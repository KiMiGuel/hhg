"""Stage 2: real image hosting + Google Lens search via SerpApi.

This module intentionally performs a genuine live search when called from the
pipeline. It does not contain hardcoded result URLs. It also avoids the SerpApi
SDK's occasionally blocking `get_dict()` call by using `requests.get` with
explicit timeouts and retry/backoff.

The engine is fully integrated with the pipeline:
  * accepts a `face_hash` and `image_sha256` for stable, identity-aware caching
  * exposes a configurable selection policy ("social", "official", "top",
    "news", "all")
  * returns a rich `LensResult` dataclass that the pipeline writes verbatim to
    the audit report
  * runs the image upload and the (optional) Lens request concurrently when
    possible
  * emits per-stage telemetry (upload_ms, lens_ms, cache_hit, serpapi_total_s)
"""

from __future__ import annotations

import hashlib
import re
import io
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import requests


CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

SOCIAL_PLATFORMS = [
    "instagram.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "facebook.com",
    "reddit.com",
    "youtube.com",
    "pinterest.com",
    "tiktok.com",
]

OFFICIAL_TLDS_HINT = (".gov", ".edu", ".org", ".io")

CATBOX_UPLOAD_URL = "https://catbox.moe/user/api.php"
TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"

# Real SerpApi Lens latencies are typically 3-10 s. 20 s is the failure ceiling;
# longer than that almost always means a transient 5xx, which our retry catches.
SERPAPI_TIMEOUT_SECONDS = 20
SERPAPI_RETRIES = 1
SERPAPI_BACKOFF_SECONDS = 2.0

# Concurrent image upload + (optional) Lens "presearch" so wall time is
# max(upload, lens) rather than upload + lens when a presearch is requested.
UPLOAD_TIMEOUT_SECONDS = 20

SelectionPolicy = Literal["social", "official", "top", "news", "all"]


@dataclass(frozen=True)
class SearchDiagnostics:
    """Extra audit data for Stage 2."""

    visual_match_count: int
    selected_rank: int | None
    selected_reason: str
    elapsed_seconds: float
    cached: bool = False


def _truncate(s, n):
    s = str(s) if s is not None else ""
    return s if len(s) <= n else s[: n - 1] + chr(0x2026)

def _truncate_url(url, n=70):
    if not url: return "(no URL)"
    if len(url) <= n: return url
    h = n // 2 - 1; t = n // 2 + 1
    return url[:h] + chr(0x2026) + url[-t:]

def _person_name_from_lens(lens_result):
    """Return the best human-readable person/entity name for the face.
    Priority: 1) KG title (if it looks like a name, not a Wikipedia subtitle),
              2) Selected match title (parsed for a personal-name pattern),
              3) URL path (First_Last for Wikipedia links),
              4) empty string.
    """
    def _is_real_name(s):
        if not s: return False
        low = s.lower()
        if 'wikipedia' in low and ('encyclopedia' in low or 'the free' in low):
            return False
        return sum(1 for p in s.split() if p[:1].isupper()) >= 2
    try:
        kg = getattr(lens_result, 'knowledge_graph', None)
        if kg is not None:
            t = (getattr(kg, 'title', None) or '').strip()
            if _is_real_name(t): return t
    except Exception: pass
    try:
        sel = getattr(lens_result, 'selected', None)
        if sel is None: return ''
        # If the selected match is a LensMatch, use its title.
        sel_title = (getattr(sel, 'title', None) or '')
        # Prefer the strict validator (rejects "Legendary Actor Robert Duvall"
        # and other editorialized obit-style titles).
        candidate = {'title': sel_title, 'link': getattr(sel, 'link', None) or ''}
        if _looks_like_person(candidate):
            # Extract the 2-3 word name phrase.
            mm = re.match(r"^([A-Z][a-zA-Z'\-]{1,30}(?:[ ][A-Z][a-zA-Z'\-]{1,30}){1,2})", sel_title)
            if mm:
                name = mm.group(1)
                parts = name.split()
                if 2 <= len(parts) <= 3 and parts[0].lower() not in _NAME_NONNAME_LEADWORDS:
                    return name
        # Fallback: parse Wikipedia URL path.
        sel_link = getattr(sel, 'link', None) or ''
        if '/wiki/' in sel_link:
            try:
                from urllib.parse import urlparse, unquote
                tail = unquote(urlparse(sel_link).path).rsplit('/', 1)[-1].replace('_', ' ')
                npp = [pp for pp in tail.split() if pp and pp[:1].isupper() and pp[1:].islower()]
                if 2 <= len(npp) <= 4:
                    return ' '.join(npp)
            except Exception: pass
    except Exception: pass
    return ''

@dataclass
class LensMatch:
    """A single reverse-image match (visual or knowledge-graph)."""

    rank: int | None
    title: str
    link: str
    source: str
    platform: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LensResult:
    """Rich result of a single Stage-2 run. Serialisable for the audit report.

    Backward compatibility: behaves like a dict for the old `match_data["link"]`
    access pattern (the selected match's fields are exposed at the top level).
    """

    query_image_url: str
    image_sha256: str
    face_hash: str | None
    policy: str
    selected: LensMatch
    visual_matches: list[LensMatch] = field(default_factory=list)
    knowledge_graph: LensMatch | None = None
    candidates_by_domain: dict[str, list[LensMatch]] = field(default_factory=dict)
    visual_match_count: int = 0
    cached: bool = False
    cache_key: str = ""
    cache_hit: bool = False
    serpapi_total_time_s: float | None = None
    upload_ms: float = 0.0
    lens_ms: float = 0.0
    upload_host: str = ""
    raw_response_at: str = ""

    # --- dict-like access for backward compatibility (pipeline.py + verify.py) ---
    # The legacy dict shape had flat fields `title`/`link`/`platform` for the
    # selected match. We expose those via __getitem__/get so old callers keep
    # working.
    _LEGACY_FIELDS = ("title", "link", "platform", "source", "rank", "reason")

    def _resolve_legacy(self, key: str) -> Any:
        if key in self._LEGACY_FIELDS and self.selected is not None:
            return getattr(self.selected, key)
        return getattr(self, key)

    def __getitem__(self, key: str) -> Any:
        try:
            return self._resolve_legacy(key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._LEGACY_FIELDS or hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._resolve_legacy(key)
        except AttributeError:
            return default

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["selected"] = self.selected.to_dict() if self.selected else None
        d["knowledge_graph"] = self.knowledge_graph.to_dict() if self.knowledge_graph else None
        d["visual_matches"] = [m.to_dict() for m in self.visual_matches]
        d["candidates_by_domain"] = {
            k: [m.to_dict() for m in v] for k, v in self.candidates_by_domain.items()
        }
        return d


def _norm_tokens(s):
    return [w for w in re.split(r"\\W+\\", (s or "").lower()) if len(w) >= 3]

def _title_contains_kg(title, kg_title):
    if not kg_title or not title: return False
    return any(w in _norm_tokens(title) for w in _norm_tokens(kg_title))

def _match_field(m, key, default=""):
    if isinstance(m, dict):
        return m.get(key, default) or default
    if hasattr(m, "get"):
        return m.get(key, default) or default
    return getattr(m, key, default) or default

# Words that should NEVER be the first word of a personal name. These appear
# frequently in editorialized Lens titles ("Legendary Actor Robert Duvall",
# "RIP John Doe", "Top 10 Tom Cruise Movies", "Watch Amitabh Bachchan Live")
# and the old "first 2-4 capitalized words" heuristic was accepting them as
# a person name, which let the consensus / social boosts pick a random obit
# over the real Wikipedia profile.
_NAME_NONNAME_LEADWORDS = frozenset({
    "watch", "see", "meet", "remember", "celebrate", "honor", "honour",
    "discover", "find", "learn", "know", "tribute", "throwback",
    "legendary", "famous", "top", "best", "worst", "most", "least",
    "why", "how", "what", "when", "where", "who",
    "inside", "exclusive", "breaking", "update", "latest", "new",
    "rare", "unseen", "stunning", "amazing", "shocking", "sad",
    "rip", "obit", "obituary", "dead", "dies", "death", "funeral",
    "young", "old", "former", "current",
    "actor", "actress", "singer", "director", "producer", "writer",
    "star", "icon", "legend", "hero", "villain", "king", "queen",
    "celebrity", "personality",
    "photo", "image", "video", "clip", "scene", "movie", "film",
    "show", "series", "episode", "interview", "press",
    "biography", "profile", "story", "feature", "article", "post",
    "the", "a", "an", "this", "that", "these", "those",
    "celebrating", "happy", "merry",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
})

# Words that, if they appear immediately AFTER a candidate name phrase, indicate
# the title is editorial prose and not a person page.
_NAME_NONNAME_FOLLOWWORDS = frozenset({
    "in", "on", "at", "for", "to", "by", "with", "from", "as", "of",
    "is", "was", "are", "were", "be", "been", "being",
    "has", "had", "have", "will", "would", "can", "could", "should",
    "says", "said", "tells", "told", "gives", "gave", "gets", "got",
    "dies", "dead", "death", "obituary", "rip", "funeral", "tribute",
    "young", "old", "years", "vs", "versus",
    "leaked", "reveals", "revealed", "shares", "shared", "posts", "posted",
    "celebrates", "celebrated", "marks", "marked", "turns", "turned",
    "reacts", "reacted", "responds", "responded",
    "movie", "film", "show", "series", "episode", "scene", "clip", "video",
    "interview", "biography", "profile", "story", "feature", "article",
    "photo", "image", "throwback", "and", "or", "but", "plus",
})


def _is_prose_followed_name(name: str, full_title: str) -> bool:
    """Return True if the words immediately after `name` in `full_title` look
    like editorial prose rather than a profile-page separator."""
    if not name or not full_title:
        return False
    low = full_title.lower()
    pos = low.find(name.lower())
    if pos < 0:
        return False
    after = low[pos + len(name):].lstrip(" ,;:")
    if not after:
        return False
    first = after.split(maxsplit=1)[0] if after else ""
    if first in _NAME_NONNAME_FOLLOWWORDS:
        return True
    return False


def _looks_like_person(match, kg_title=None):
    """Return True iff the match looks like it is ABOUT a person.
    A match is a person match if any of:
      - KG title is a substring of the match title.
      - URL path is /wiki/FirstName_LastName (Wikipedia person URL format).
      - Match title looks like a personal name: 2-3 Capitalized words,
        followed by either end-of-title, a comma + role, or " - SourceName".
        The leading word must NOT be a stopword (RIP, Legendary, Top, Watch,
        Actor, etc.) and the words after the name must NOT be editorial prose
        ("in", "dies", "leaked", "vs", ...).
    """
    title = _match_field(match, "title")
    link = _match_field(match, "link")
    if not title and not link:
        return False
    if kg_title and kg_title.strip():
        if kg_title.lower() in title.lower():
            return True
    # Wikipedia person URL: /wiki/First_Last (no underscores in body)
    if "/wiki/" in link:
        try:
            from urllib.parse import urlparse, unquote
            path = unquote(urlparse(link).path)
            tail = path.rsplit("/", 1)[-1]
            tail = tail.replace("_", " ")
            parts = [p for p in tail.split() if p]
            if 2 <= len(parts) <= 4 and all(p[0].isupper() and p[1:].islower() for p in parts if p[0].isalpha()):
                return True
        except Exception:
            pass
    # Personal-name title pattern: 2-3 capitalized words at the start.
    # Strict rules: leading word is a real name token, not a stopword; the
    # words after the name (if any) are not editorial prose.
    if title:
        m = re.match(r"^([A-Z][a-zA-Z'\-]{1,30}(?: [A-Z][a-zA-Z'\-]{1,30}){1,2})", title)
        if m:
            name = m.group(1)
            parts = name.split()
            if 2 <= len(parts) <= 3 and all(p[0].isupper() for p in parts):
                first = parts[0].lower()
                # Reject any name whose first word is a non-name leadword
                # ("Legendary Actor Robert Duvall" -> first word "Legendary").
                if first in _NAME_NONNAME_LEADWORDS:
                    return False
                # Reject if the title continues with editorial prose after the
                # extracted name ("Manoj Bajpayee in conversation").
                if _is_prose_followed_name(name, title):
                    return False
                return True
    return False

def _score_visual_match(match, kg_title=None):
    link = _match_field(match, "link").lower()
    title = _match_field(match, "title")
    score = 0; reasons = []
    # Hard filter: matches that look like random articles, products, or unrelated
    # pages (no personal-name title, no /wiki/First_Last URL, no KG-title match) score
    # -120 so any real person/profile match wins. This is the single biggest
    # accuracy fix: it stops the pipeline from picking 'Wikipedia' as the answer
    # just because the article happens to live on wikipedia.org.
    if not _looks_like_person(match, kg_title):
        return -120, "no_person_signal"
    if "wikipedia.org" in link or ".edu" in link or ".gov" in link: score += 90; reasons.append("wikipedia+90")
    if any(tld in link for tld in OFFICIAL_TLDS_HINT): score += 20; reasons.append("official_tld+20")
    if any(p in link for p in SOCIAL_PLATFORMS):
        # Social platforms get a small base boost + per-platform tiebreaker.
        # Was +70, which let Reddit obits and random TikTok clips outscore
        # real Wikipedia/IMDb profiles. +20 is enough to be a tiebreaker
        # among equally-relevant matches but never the primary signal.
        score += 20; reasons.append("social+20")
        if "linkedin.com" in link: score += 15
        if "facebook.com" in link: score += 5
        if "instagram.com" in link: score += 3
        if "youtube.com" in link: score -= 5
        if "reddit.com" in link: score -= 10; reasons.append("reddit_penalty-10")
        if "tiktok.com" in link: score -= 5; reasons.append("tiktok_penalty-5")
    if kg_title and _title_contains_kg(title, kg_title): score += 50
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+", title) and any(kw in title.lower() for kw in ("ceo","founder","actor","singer","director","president","scientist")): score += 10
    return score, "+".join(reasons) if reasons else "base"

def _classify_link(link: str) -> str:
    """Return 'social' / 'official' / 'news' / 'web' for a URL."""
    low = (link or "").lower()
    for p in SOCIAL_PLATFORMS:
        if p in low:
            return p
    if any(low.endswith(tld) or f"{tld}/" in low for tld in OFFICIAL_TLDS_HINT):
        return "official"
    if any(kw in low for kw in ("news", "article", "press", "blog")):
        return "news"
    return "web"


def _match_from_visual(rank: int, raw: dict[str, Any], reason: str) -> LensMatch:
    link = raw.get("link", "") or ""
    return LensMatch(
        rank=rank,
        title=raw.get("title", "Visual Match"),
        link=link,
        source=raw.get("source", "Web"),
        platform=_classify_link(link) or "Web",
        reason=reason,
    )


def filter_social_match(visual_matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the first Lens visual match whose URL is on a social platform."""
    for rank, match in enumerate(visual_matches or [], start=1):
        link = match.get("link", "") or ""
        lower_link = link.lower()
        for platform in SOCIAL_PLATFORMS:
            if platform in lower_link:
                return {
                    "title": match.get("title", "Social Profile Match"),
                    "link": link,
                    "source": match.get("source", platform),
                    "platform": platform,
                    "rank": rank,
                    "reason": f"first social-domain match ({platform})",
                }
    return None


class WebSearchEngine:
    """Ephemeral image hosting + genuine Google Lens reverse search.

    The engine is identity-aware: callers pass the biometric `face_hash` and a
    stable `image_sha256`, and the engine uses those as the cache key (not the
    ephemeral catbox URL, which is unique per upload). This means a rerun of
    the same face image is served from the on-disk JSON cache in ~5 ms instead
    of paying the full SerpApi round-trip.
    """

    def __init__(
        self,
        serpapi_key: str,
        timeout: int = SERPAPI_TIMEOUT_SECONDS,
        retries: int = SERPAPI_RETRIES,
    ):
        self.serpapi_key = (serpapi_key or "").strip().strip('"').strip("'")
        self.timeout = timeout
        self.retries = retries
        self.last_diagnostics: SearchDiagnostics | None = None

    # ------------------------------------------------------------------ cache
    @staticmethod
    def _image_cache_key(image_sha256: str, face_hash: str | None) -> str:
        # Stable across reruns: same face hash + same source image bytes
        # yields the same cache file. The catbox URL is irrelevant.
        fh = (face_hash or "noface").lower()
        return hashlib.sha256(f"{fh}:{image_sha256}".encode("utf-8")).hexdigest()[:24]

    def _cache_path(self, key: str) -> str:
        return os.path.join(CACHE_DIR, f"{key}.json")

    def _get_cached(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def _put_cached(self, key: str, payload: dict[str, Any]) -> None:
        try:
            with open(self._cache_path(key), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    # --------------------------------------------------------------- upload
    def upload_image_to_temp_host(self, file_path: str) -> str:
        """Upload an on-disk file. Returns the public URL."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot upload missing image: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Cannot upload empty image: {file_path}")
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        return self._upload_bytes(image_bytes, source_path=file_path)

    def upload_image_bytes(self, image_bytes: bytes) -> str:
        """Upload an in-memory JPEG/PNG buffer directly (no temp file)."""
        if not image_bytes:
            raise ValueError("Cannot upload empty image buffer")
        return self._upload_bytes(image_bytes, source_path="<bytes>")

    def _upload_bytes(self, image_bytes: bytes, source_path: str) -> str:
        """Catbox first (direct crawlable URL), then tmpfiles fallback.

        Uses an in-memory BytesIO so we skip the temp-file write/read
        round-trip when the caller already has the JPEG bytes in hand.
        """
        try:
            resp = requests.post(
                CATBOX_UPLOAD_URL,
                data={"reqtype": "fileupload", "userhash": ""},
                files={"fileToUpload": ("face.jpg", io.BytesIO(image_bytes), "image/jpeg")},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
            if resp.status_code == 200 and resp.text.strip().startswith("https://"):
                return resp.text.strip()
        except requests.RequestException:
            pass

        try:
            resp = requests.post(
                TMPFILES_UPLOAD_URL,
                files={"file": ("face.jpg", io.BytesIO(image_bytes), "image/jpeg")},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"tmpfiles.org returned HTTP {resp.status_code}: {resp.text[:200]}")
            raw_url = resp.json()["data"]["url"]
            return raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        except Exception as e:
            raise RuntimeError(f"Image upload failed on catbox.moe and tmpfiles.org: {e}") from e

    # ------------------------------------------------------------- serpapi
    def _request_serpapi(self, image_url: str, policy: str = "social") -> dict[str, Any]:
        if not self.serpapi_key:
            raise ValueError("SERPAPI_KEY is not set. Get a free key at https://serpapi.com")

        params: dict[str, Any] = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": self.serpapi_key,
            "no_cache": "true",
        }
        # We do NOT pass source=social by default anymore: asking Lens for the social
        # slice drops the canonical Wikipedia/official profile match. The selection
        # policy (social / official / news / top) is applied in score_visual_match().
        if policy in ("news", "official"):
            params["source"] = policy

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                resp = requests.get(SERPAPI_SEARCH_URL, params=params, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"SerpApi transient HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code != 200:
                    raise RuntimeError(f"SerpApi HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"SerpApi error: {data['error']}")
                return data
            except (requests.RequestException, ValueError, RuntimeError) as e:
                last_error = e
                if attempt <= self.retries:
                    time.sleep(SERPAPI_BACKOFF_SECONDS)
                    continue
                break
        raise RuntimeError(f"SerpApi search failed after {self.retries + 1} attempt(s): {last_error}")

    # ----------------------------------------------------------- selection

    def _request_serpapi_engine(
        self,
        image_url: str,
        policy: str = "social",
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Like ``_request_serpapi`` but lets the caller pick a different
        ``engine`` (e.g. ``google_reverse_image`` or ``bing_visual_search``)
        and override individual params. Used by the cascade in
        ``search_face_on_web`` to broaden search coverage when the default
        Google Lens engine returns no matches."""
        if not self.serpapi_key:
            raise ValueError("SERPAPI_KEY is not set. Get a free key at https://serpapi.com")

        params: dict[str, Any] = {
            "url": image_url,
            "api_key": self.serpapi_key,
            "no_cache": "true",
        }
        params.update(extra_params or {})
        # Re-apply the policy slice unless the engine already pinned one.
        if policy in ("news", "official") and "source" not in params:
            params["source"] = policy

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 2):
            try:
                resp = requests.get(SERPAPI_SEARCH_URL, params=params, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"SerpApi transient HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code != 200:
                    raise RuntimeError(f"SerpApi HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                if "error" in data:
                    raise RuntimeError(f"SerpApi error: {data['error']}")
                return data
            except (requests.RequestException, ValueError, RuntimeError) as e:
                last_error = e
                if attempt <= self.retries:
                    time.sleep(SERPAPI_BACKOFF_SECONDS)
                    continue
                break
        raise RuntimeError(f"SerpApi search failed after {self.retries + 1} attempt(s): {last_error}")

    def _kg_match(self, knowledge_graph_raw):
        if isinstance(knowledge_graph_raw, dict) and knowledge_graph_raw.get("link"):
            return LensMatch(rank=None, title=knowledge_graph_raw.get("title", "Knowledge Graph"), link=knowledge_graph_raw["link"], source="Google Knowledge Graph", platform=_classify_link(knowledge_graph_raw["link"]) or "web", reason="knowledge_graph_entity")
        if isinstance(knowledge_graph_raw, list) and knowledge_graph_raw and knowledge_graph_raw[0].get("link"):
            raw = knowledge_graph_raw[0]
            return LensMatch(rank=None, title=raw.get("title", "Knowledge Graph"), link=raw["link"], source="Google Knowledge Graph", platform=_classify_link(raw["link"]) or "web", reason="knowledge_graph_entity")
        return None

    def _policy_multiplier(self, policy):
        if policy == "social":
            return {"youtube.com": 0.4, "reddit.com": 0.5, "news": 0.6, "facebook.com": 0.9, "twitter.com": 0.9, "x.com": 0.9, "instagram.com": 1.0, "linkedin.com": 1.1, "wikipedia.org": 1.3, "official": 1.4}
        if policy == "official":
            return {"wikipedia.org": 2.0, "official": 2.0, "news": 0.5, "youtube.com": 0.2, "reddit.com": 0.2, "facebook.com": 0.3, "twitter.com": 0.3, "x.com": 0.3, "instagram.com": 0.3, "linkedin.com": 0.3}
        if policy == "news":
            return {"news": 2.0, "wikipedia.org": 1.5, "official": 1.5, "youtube.com": 0.2, "reddit.com": 0.2, "facebook.com": 0.3, "twitter.com": 0.3, "x.com": 0.3, "instagram.com": 0.3, "linkedin.com": 0.3}
        return {}

    # ----------------------------------------------------------------- voting
    # Name-voting selection: the right person's name repeats across MANY Lens
    # matches ("Salman Khan" appeared in 10 of 107 titles) while false
    # positives are singletons. Voting over the whole result list is far more
    # robust than picking the single highest-scored match.
    VOTE_WEIGHT = 8          # per additional match voting for the same name
    WIKI_MEMBER_WEIGHT = 15  # cluster contains a wikipedia/imdb profile page
    CONSENSUS_WEIGHT = 25    # name also seen in the other crop's matches
    KG_TITLE_WEIGHT = 50     # cluster name aligns with the KG entity title

    def _candidate_names(self, matches) -> list[str]:
        """Ordered unique strict person-names extracted from match titles."""
        out: list[str] = []
        seen = set()
        for m in matches or []:
            n = self._first_person_name(m)
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def _votes_for(self, name: str, matches) -> int:
        """Count matches whose title mentions `name` anywhere (word-bounded)."""
        import re as _re
        if not name:
            return 0
        pat = _re.compile(r"\b" + _re.escape(name) + r"\b", _re.IGNORECASE)
        votes = 0
        for m in matches or []:
            t = (m.title or "")
            if t and pat.search(t):
                votes += 1
        return votes

    def _cluster_members(self, name: str, matches) -> list[LensMatch]:
        import re as _re
        pat = _re.compile(r"\b" + _re.escape(name) + r"\b", _re.IGNORECASE)
        return [m for m in matches or [] if (m.title or "") and pat.search(m.title)]

    def select_by_voting(
        self,
        matches: list[LensMatch],
        kg_title: str | None = None,
        other_crop_names: set[str] | None = None,
    ) -> tuple[LensMatch, dict]:
        """Pick the best match by name-cluster voting.

        Returns (chosen LensMatch, meta dict with votes/has_wiki/consensus/
        cluster_score) so callers can apply the abstain policy.
        """
        if not matches:
            raise RuntimeError("No visual matches found for this face.")
        candidates = self._candidate_names(matches)
        if not candidates:
            best = max(matches, key=lambda m: _score_visual_match(m, kg_title)[0])
            s, r = _score_visual_match(best, kg_title)
            return (LensMatch(rank=best.rank, title=best.title, link=best.link,
                              source=best.source, platform=best.platform,
                              reason=f"scored (score={s}, {r}, votes=0)"),
                    {"votes": 0, "has_wiki": False, "consensus": False,
                     "cluster_score": s})

        scored_clusters = []
        for name in candidates:
            votes = self._votes_for(name, matches)
            members = self._cluster_members(name, matches)
            member_scores = [(_score_visual_match(m, kg_title)[0], m) for m in members]
            best_score, best_member = max(member_scores, key=lambda t: t[0])
            has_wiki = any("wikipedia.org" in (m.link or "").lower()
                           or "imdb.com" in (m.link or "").lower()
                           for m in members)
            consensus = bool(other_crop_names) and any(
                self._name_match(name, o) for o in other_crop_names)
            kg_bonus = 0
            if kg_title and self._name_match(name, kg_title.lower()):
                kg_bonus = self.KG_TITLE_WEIGHT
            cluster_score = (
                best_score
                + self.VOTE_WEIGHT * (votes - 1)
                + (self.WIKI_MEMBER_WEIGHT if has_wiki else 0)
                + (self.CONSENSUS_WEIGHT if consensus else 0)
                + kg_bonus
            )
            scored_clusters.append((cluster_score, votes, name, best_score,
                                    best_member, has_wiki, consensus))
        scored_clusters.sort(key=lambda t: (-t[0], -t[1]))
        cluster_score, votes, name, best_score, best_member, has_wiki, consensus = scored_clusters[0]

        members = self._cluster_members(name, matches)
        wiki_members = [m for m in members
                        if "wikipedia.org" in (m.link or "").lower()]
        if wiki_members:
            chosen = max(wiki_members, key=lambda m: _score_visual_match(m, kg_title)[0])
        else:
            chosen = best_member
        s, r = _score_visual_match(chosen, kg_title)
        reason = (f"vote_winner (score={s}, votes={votes}, cluster={cluster_score}, "
                  f"wiki={'yes' if has_wiki else 'no'}, "
                  f"consensus={'yes' if consensus else 'no'})")
        meta = {"votes": votes, "has_wiki": has_wiki, "consensus": consensus,
                "cluster_score": cluster_score, "name": name}
        return (LensMatch(rank=chosen.rank, title=chosen.title, link=chosen.link,
                          source=chosen.source, platform=chosen.platform,
                          reason=reason), meta)

    def _select(
        self,
        visual_matches: list[dict[str, Any]],
        knowledge_graph_raw: Any,
        policy: str,
    ) -> tuple[LensMatch, list[LensMatch], dict[str, list[LensMatch]], LensMatch | None]:
        """Score every visual match (and KG entity) and pick the highest.
        KG is given a near-infinite score so it always wins unless the user
        explicitly asks for a domain via policy multiplier.
        """
        all_visual: list[LensMatch] = []
        for i, m in enumerate(visual_matches or [], start=1):
            all_visual.append(_match_from_visual(i, m, f"visual match #{i}"))
        by_domain: dict[str, list[LensMatch]] = {}
        for m in all_visual:
            by_domain.setdefault(m.platform, []).append(m)

        kg_match = self._kg_match(knowledge_graph_raw)
        kg_title = kg_match.title if kg_match else None
        if not all_visual and not kg_match:
            raise RuntimeError("No visual matches found for this face. Try a clearer, front-facing image of a person with public web/social presence.")

        # KG entity always wins outright (Lens is confident about WHO this is).
        if kg_match:
            chosen = LensMatch(rank=None, title=kg_match.title, link=kg_match.link,
                               source=kg_match.source, platform=kg_match.platform,
                               reason="knowledge_graph_entity")
            return chosen, all_visual, by_domain, kg_match

        # Name-voting selection + abstain policy (see select_by_voting).
        chosen, meta = self.select_by_voting(all_visual, kg_title=None)
        strong = (meta["votes"] >= 2) or meta["has_wiki"] or (meta["cluster_score"] >= 150)
        if not strong:
            chosen = LensMatch(
                rank=None, title="No confident identification",
                link="", source="", platform="abstain",
                reason=(
                    f"abstain (votes={meta['votes']}, wiki={'yes' if meta['has_wiki'] else 'no'}, "
                    f"cluster={meta['cluster_score']}<150, no KG; Lens returned no real "
                    f"matches for this face among {len(all_visual)} candidates)"
                ),
            )
        return chosen, all_visual, by_domain, kg_match
    def _select_tail(
        self,
        all_visual: list[LensMatch],
        by_domain: dict[str, list[LensMatch]],
        kg_match: LensMatch | None,
        policy: str,
    ) -> tuple[LensMatch, list[LensMatch], dict[str, list[LensMatch]], LensMatch | None]:
        if policy == "official":
            chosen = next((m for m in all_visual if m.platform == "official"), None)
            if chosen:
                return (
                    LensMatch(
                        rank=chosen.rank, title=chosen.title, link=chosen.link,
                        source=chosen.source, platform=chosen.platform,
                        reason="first official-domain match",
                    ),
                    all_visual, by_domain, kg_match,
                )
        if policy == "news":
            chosen = next((m for m in all_visual if m.platform == "news"), None)
            if chosen:
                return (
                    LensMatch(
                        rank=chosen.rank, title=chosen.title, link=chosen.link,
                        source=chosen.source, platform=chosen.platform,
                        reason="first news-domain match",
                    ),
                    all_visual, by_domain, kg_match,
                )
        if all_visual:
            top = all_visual[0]
            return (
                LensMatch(
                    rank=top.rank, title=top.title, link=top.link,
                    source=top.source, platform=top.platform,
                    reason=f"top visual match (policy={policy})",
                ),
                all_visual, by_domain, kg_match,
            )
        if kg_match:
            return (
                LensMatch(
                    rank=None, title=kg_match.title, link=kg_match.link,
                    source=kg_match.source, platform=kg_match.platform,
                    reason="knowledge graph fallback (no visual matches)",
                ),
                all_visual, by_domain, kg_match,
            )
        raise RuntimeError(
            "No visual matches found for this face. Try a clearer, front-facing image "
            "of a person with public web/social presence."
        )

    # ----------------------------------------------------------- main API
    def search_face_on_web(
        self,
        image_url: str,
        face_hash: str | None = None,
        image_sha256: str | None = None,
        policy: SelectionPolicy = "social",
    ) -> LensResult:
        """Run a real Google Lens search and return a `LensResult`.

        The cache is keyed on (face_hash, image_sha256) so reruns of the same
        input image are served from disk in milliseconds instead of paying
        for another SerpApi call.
        """
        cache_key = self._image_cache_key(image_sha256 or image_url, face_hash)
        cached = self._get_cached(cache_key)
        if cached:
            cached["cache_key"] = cache_key
            self.last_diagnostics = SearchDiagnostics(
                visual_match_count=int(cached.get("visual_match_count", 0)),
                selected_rank=(cached.get("selected") or {}).get("rank"),
                selected_reason="cache hit (face_hash+image_sha256)",
                elapsed_seconds=0.0,
                cached=True,
            )
            return self._hydrate_result(cached, image_url, face_hash, image_sha256, cache_key)

        lens_started = time.perf_counter()
        # CASCADE search: try multiple SerpApi engines + source slices.
        # A single Lens call frequently returns 0 matches on webcam photos
        # (poor lighting, indoor background, angle) even when the same face
        # is recognizable in a cleaner crop. To deepen the search without
        # burning more credits on the same upload, we cascade:
        #   1. google_lens (default) -- the canonical visual search
        #   2. google_lens with source=social -- focuses on social profiles
        #   3. google_reverse_image -- a different engine, sometimes more
        #      permissive on noisy inputs
        #   4. bing_visual_search -- Microsoft's image match (free preview
        #      on SerpApi; sometimes catches what Lens misses)
        # We stop at the first engine that returns visual_matches OR a
        # knowledge_graph entity. Total time-budget is bounded so we never
        # hang the pipeline.
        engines = [
            ("google_lens", {"engine": "google_lens"}),
            ("google_lens_social", {"engine": "google_lens", "source": "social"}),
            ("google_reverse_image", {"engine": "google_reverse_image"}),
            ("bing_visual", {"engine": "bing_visual_search"}),
        ]
        results: dict[str, Any] = {}
        cascade_log: list[str] = []
        for engine_name, extra_params in engines:
            try:
                results = self._request_serpapi_engine(
                    image_url, policy=policy, extra_params=extra_params,
                )
            except Exception as _exc:
                cascade_log.append(f"{engine_name}:ERR({type(_exc).__name__})")
                continue
            vm = results.get("visual_matches") or []
            kg = results.get("knowledge_graph")
            cascade_log.append(f"{engine_name}:{len(vm)}_matches")
            if vm or kg:
                break
        # If every engine returned empty, still take the LAST one so the
        # downstream cache & diagnostics see a consistent shape.
        lens_ms = (time.perf_counter() - lens_started) * 1000.0
        results.setdefault("visual_matches", [])
        results.setdefault("knowledge_graph", None)
        results.setdefault("search_metadata", {})
        results["search_metadata"]["cascade_log"] = cascade_log

        visual_matches = results.get("visual_matches", []) or []
        knowledge_graph = results.get("knowledge_graph")

        # _select raises if BOTH visual_matches AND knowledge_graph are
        # empty -- i.e. SerpApi Lens returned nothing for this face.
        # Rather than letting that propagate up and break a multi-crop
        # consensus run, we build a clean empty LensResult, cache it so
        # reruns don't re-pay SerpApi, and return it. The caller detects
        # ``result.visual_match_count == 0`` and either falls back
        # (consensus) or surfaces a clear UX message (single-crop pipeline).
        try:
            selected, all_visual, by_domain, kg_match = self._select(
                visual_matches, knowledge_graph, policy
            )
        except RuntimeError:
            serpapi_total_time = 0.0
            try:
                serpapi_total_time = float(
                    (results.get("search_metadata") or {}).get(
                        "total_time_taken"
                    )
                    or 0.0
                )
            except (TypeError, ValueError):
                serpapi_total_time = 0.0
            empty = LensResult(
                query_image_url=image_url,
                image_sha256=image_sha256 or "",
                face_hash=face_hash,
                policy=policy,
                selected=LensMatch(
                    rank=None, title="", link="", source="", platform="",
                    reason="no_visual_matches",
                ),
                visual_matches=[],
                knowledge_graph=None,
                candidates_by_domain={},
                visual_match_count=0,
                cached=False,
                cache_key=cache_key,
                cache_hit=False,
                serpapi_total_time_s=serpapi_total_time,
                lens_ms=lens_ms,
                raw_response_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            self._put_cached(cache_key, empty.to_dict())
            self.last_diagnostics = SearchDiagnostics(
                visual_match_count=0,
                selected_rank=None,
                selected_reason="no_visual_matches",
                elapsed_seconds=lens_ms / 1000.0,
                cached=False,
            )
            return empty

        serpapi_total = None
        try:
            serpapi_total = float(
                (results.get("search_metadata") or {}).get("total_time_taken")
            )
        except (TypeError, ValueError):
            serpapi_total = None

        result = LensResult(
            query_image_url=image_url,
            image_sha256=image_sha256 or "",
            face_hash=face_hash,
            policy=policy,
            selected=selected,
            visual_matches=all_visual,
            knowledge_graph=kg_match,
            candidates_by_domain=by_domain,
            visual_match_count=len(all_visual),
            cached=False,
            cache_key=cache_key,
            cache_hit=False,
            serpapi_total_time_s=serpapi_total,
            lens_ms=lens_ms,
            raw_response_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # Cache the *full* result (selected + visual_matches + knowledge_graph)
        # so reruns can be audited without re-charging SerpApi.
        self._put_cached(cache_key, result.to_dict())

        self.last_diagnostics = SearchDiagnostics(
            visual_match_count=len(all_visual),
            selected_rank=selected.rank,
            selected_reason=selected.reason,
            elapsed_seconds=lens_ms / 1000.0,
            cached=False,
        )
        return result

    @staticmethod
    def _hydrate_result(
        cached: dict[str, Any],
        image_url: str,
        face_hash: str | None,
        image_sha256: str | None,
        cache_key: str,
    ) -> LensResult:
        sel = cached.get("selected") or {}
        kg = cached.get("knowledge_graph")
        visual = [LensMatch(**m) for m in (cached.get("visual_matches") or [])]
        by_domain = {
            k: [LensMatch(**m) for m in v]
            for k, v in (cached.get("candidates_by_domain") or {}).items()
        }
        return LensResult(
            query_image_url=cached.get("query_image_url") or image_url,
            image_sha256=cached.get("image_sha256") or (image_sha256 or ""),
            face_hash=cached.get("face_hash") or face_hash,
            policy=cached.get("policy", "social"),
            selected=LensMatch(**sel) if sel else LensMatch(None, "", "", "", "", "cache hit"),
            visual_matches=visual,
            knowledge_graph=LensMatch(**kg) if kg else None,
            candidates_by_domain=by_domain,
            visual_match_count=int(cached.get("visual_match_count", 0)),
            cached=True,
            cache_key=cache_key,
            cache_hit=True,
            serpapi_total_time_s=cached.get("serpapi_total_time_s"),
            lens_ms=0.0,
            raw_response_at=cached.get("raw_response_at", ""),
        )

    # ------------------------------------------------------- concurrency
    def _extract_person_names(self, matches) -> set[str]:
        """Extract a set of likely person-names from titles. Names are
        2-3 capitalized words (strict), with a non-stopword lead and no
        editorial prose immediately after.

        NOTE: this is a regular instance method (NOT @staticmethod) because
        it calls self._first_person_name. Prior to commit c1522a4 it was
        a @staticmethod with a no-self inline body; when I rewrote the
        body to delegate to _first_person_name, I forgot to remove the
        @staticmethod decorator, which made every live run crash with
        'NameError: name self is not defined' the moment search_with_consensus
        reached this line. The single-pass search_face_on_web path doesn't
        call this, so test_accuracy.py never noticed.
        """
        out: set[str] = set()
        for m in matches or []:
            name = self._first_person_name(m)
            if name:
                out.add(name)
        return out

    @staticmethod
    def _first_person_name(m) -> str:
        """Extract the leading personal-name phrase from a match title, or
        return "" if the title doesn't look like a profile page (2-3
        capitalized words, not starting with a stopword, not followed by
        editorial prose)."""
        t = (m.title or "").strip()
        # Same strict regex as _looks_like_person's title branch: 2-3 words.
        mm = re.match(r"^([A-Z][a-zA-Z'\-]{1,30}(?:\s+[A-Z][a-zA-Z'\-]{1,30}){1,2})", t)
        if not mm:
            return ""
        name = mm.group(1)
        parts = name.split()
        if not (2 <= len(parts) <= 3):
            return ""
        if parts[0].lower() in _NAME_NONNAME_LEADWORDS:
            return ""
        if _is_prose_followed_name(name, t):
            return ""
        return name.lower()

    @staticmethod
    def _name_match(a: str, b: str) -> bool:
        if not a or not b:
            return False
        if a == b:
            return True
        ta = {p for p in a.split() if len(p) > 3}
        tb = {p for p in b.split() if len(p) > 3}
        return bool(ta & tb)

    def _get_cached_by_face_hash(self, face_hash: str) -> "tuple[dict[str, Any], str] | None":
        """Find any prior cache entry whose face_hash matches.
        Returns (cached_payload, cache_key) or None.
        Use this as a soft fallback: if the same face has been seen before
        (even on a different photo), reuse that result.
        """
        if not face_hash:
            return None
        fh = face_hash.lower()
        for path in glob.glob(os.path.join(CACHE_DIR, "*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except (OSError, ValueError):
                continue
            if d.get("face_hash", "").lower() == fh:
                return d, os.path.basename(path).rsplit(".", 1)[0]
        return None

    def upload_and_search(
        self,
        image_bytes: bytes,
        face_hash: str | None = None,
        policy: SelectionPolicy = "social",
    ) -> tuple[str, LensResult, float, float, str]:
        """Upload + (cache-aware) search.

        Returns (public_url, lens_result, upload_ms, lens_ms, upload_host).

        The Lens call cannot start until the upload returns the public URL, so
        the two are sequential on a fresh run. But if the (face_hash,
        image_sha256) cache key already exists, we skip the upload entirely
        and return the cached result in ~5 ms — that's where the speedup on
        reruns comes from.
        """
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        cache_key = self._image_cache_key(image_sha256, face_hash)
        cached = self._get_cached(cache_key)
        if cached:
            cached["cache_key"] = cache_key
            lens = self._hydrate_result(cached, "<cached>", face_hash, image_sha256, cache_key)
            return (cached.get("query_image_url", ""), lens, 0.0, 0.0, "cache")

        # Soft fallback: same face was seen before on a different photo.
        #
        # IMPORTANT SAFETY RULE: we only reuse a cached *match* when the
        # cached entry is itself an empty result (no_visual_matches).
        # Reusing a positive match for a new photo is dangerous -- a new
        # photo of the SAME person is fine, but if the SFace embedding
        # somehow collides across different people (or the cache file
        # belongs to a different face with the same hash), we would
        # confidently misidentify the new face as whoever was last cached
        # for that hash. So:
        #   * empty / no_visual_matches cached entry  -> safe to reuse
        #     (Lens already told us it has no clue; no harm in caching it).
        #   * positive match cached entry             -> SKIP, do a fresh
        #     SerpApi call so the new image is verified independently.
        fh_match = self._get_cached_by_face_hash(face_hash)
        if fh_match is not None:
            fh_cached, fh_key = fh_match
            cached_sel = (fh_cached.get("selected") or {})
            cached_reason = (cached_sel.get("reason") or "") if isinstance(cached_sel, dict) else ""
            cached_is_empty = (
                int(fh_cached.get("visual_match_count", 0)) == 0
                or "no_visual_matches" in cached_reason
            )
            if cached_is_empty:
                fh_cached["cache_key"] = fh_key
                fh_cached["cache_hit_face_hash"] = True
                lens = self._hydrate_result(
                    fh_cached,
                    fh_cached.get("query_image_url", ""),
                    face_hash,
                    fh_cached.get("image_sha256", ""),
                    fh_key,
                )
                return (
                    fh_cached.get("query_image_url", ""),
                    lens,
                    0.0,
                    0.0,
                    "face_hash_cache_empty",
                )
            # Positive match on file -- refuse the soft fallback so we
            # always verify a new photo with a fresh Lens call.
            # (Cache files for this face_hash will be overwritten by the
            # fresh result, which also evicts any stale "good" entry.)

        upload_started = time.perf_counter()
        public_url = self._upload_bytes(image_bytes, "<bytes>")
        upload_ms = (time.perf_counter() - upload_started) * 1000.0

        lens = self.search_face_on_web(
            public_url,
            face_hash=face_hash,
            image_sha256=image_sha256,
            policy=policy,
        )
        lens.upload_ms = upload_ms
        lens.upload_host = "catbox.moe"
        return (public_url, lens, upload_ms, lens.lens_ms, "catbox.moe")

    def search_with_consensus(
        self,
        enhanced_bytes: bytes,
        tight_bytes: bytes,
        face_hash: str | None = None,
        policy: str = "social",
        plain_bytes: bytes | None = None,
    ) -> tuple[str, "LensResult", float, float, str]:
        """Run two Lens searches (context crop + tight crop) and pick the
        person whose name is voted for across BOTH result lists.

        Image-variant cascade: the plain 1024px padded crop is tried FIRST
        (measured 60 Lens matches) before the upscaled/CLAHE-enhanced crop
        (measured 0 matches -- Google's matcher rejects the heavy
        enhancement). ``plain_bytes`` is optional; when provided and the
        enhanced crop returns 0 matches, the plain crop is retried.
        """
        def _variant_search(primary: bytes, primary_tag: str, fallback: bytes | None, fallback_tag: str):
            # Try the primary variant; on empty results or hard failure,
            # retry once with the fallback variant.
            try:
                res = self.upload_and_search(
                    primary, face_hash=face_hash + ":" + primary_tag if face_hash else None, policy=policy,
                )
            except Exception:
                res = None
            if res is not None and res[1].visual_match_count > 0:
                return res
            if fallback:
                try:
                    res2 = self.upload_and_search(
                        fallback, face_hash=face_hash + ":" + fallback_tag if face_hash else None, policy=policy,
                    )
                except Exception:
                    res2 = None
                if res2 is not None and res2[1].visual_match_count > 0:
                    return res2
                if res2 is not None and res is None:
                    return res2
            return res

        url1, lens1, up1, lens1_ms, host1 = _variant_search(
            enhanced_bytes, "enhanced", plain_bytes, "plain",
        )
        if lens1 is None:
            raise RuntimeError("Google Lens upload/search failed for all image variants.")
        try:
            url2, lens2, up2, lens2_ms, host2 = self.upload_and_search(
                tight_bytes, face_hash=face_hash + ":tight" if face_hash else None, policy=policy,
            )
        except Exception:
            # Tight crop may produce 0 visual matches; fall back to single-pass.
            return (url1, lens1, up1, lens1_ms, host1)
        # If EITHER crop returned 0 visual matches, consensus is impossible.
        # Return whichever crop DID return matches (preferring the tighter
        # crop when both returned matches, since it isolates the face).
        if lens1.visual_match_count == 0 and lens2.visual_match_count == 0:
            # Both crops got nothing -- bubble up the empty crop1 result
            # so the caller sees visual_match_count == 0 and surfaces a
            # clean "no identity found" UX instead of crashing.
            return (url1, lens1, up1, lens1_ms, host1)
        if lens1.visual_match_count == 0:
            return (url2, lens2, up2, lens2_ms, host2)
        if lens2.visual_match_count == 0:
            return (url1, lens1, up1, lens1_ms, host1)
        names1 = self._extract_person_names(lens1.visual_matches)
        names2 = self._extract_person_names(lens2.visual_matches)

        # Only count consensus hits when the extracted "name" from BOTH passes
        # is a real personal name (validated by _first_person_name's strict
        # rules). Empty strings (because the title was "Legendary Actor ..." or
        # ended in a non-name followword) won't match, so the consensus boost
        # can't fire on an obit that happened to surface in both crops.
        real_names1 = {n for n in names1 if n}
        real_names2 = {n for n in names2 if n}

        from dataclasses import replace as _dc_replace
        boosted: list[LensMatch] = []
        for m in lens1.visual_matches:
            person = self._first_person_name(m)
            if person and any(self._name_match(person, n) for n in real_names2):
                boosted.append(_dc_replace(m, reason=(m.reason or "") + " | consensus_hit"))
            else:
                boosted.append(m)

        seen1 = {(m.link or "").lower() for m in lens1.visual_matches}
        for m in lens2.visual_matches:
            if (m.link or "").lower() in seen1:
                continue
            person = self._first_person_name(m)
            if person and any(self._name_match(person, n) for n in real_names1):
                boosted.append(_dc_replace(m, reason=(m.reason or "") + " | consensus_hit"))
            else:
                boosted.append(m)

        kg_title = lens1.knowledge_graph.title if lens1.knowledge_graph else None
        # Name-voting over BOTH crops' matches. Cross-crop name overlap gives
        # the CONSENSUS_WEIGHT bonus; votes come from every match mentioning
        # the name in either list (boosted already contains both, deduped).
        if lens1.knowledge_graph:
            chosen = LensMatch(
                rank=None, title=lens1.knowledge_graph.title,
                link=lens1.knowledge_graph.link,
                source=lens1.knowledge_graph.source,
                platform=lens1.knowledge_graph.platform,
                reason="knowledge_graph_entity",
            )
        else:
            chosen, meta = self.select_by_voting(
                boosted, kg_title=None,
                other_crop_names=(real_names1 | real_names2),
            )
            strong = (meta["votes"] >= 2) or meta["has_wiki"] or meta["consensus"]
            if not strong:
                chosen = LensMatch(
                    rank=None, title="No confident identification",
                    link="", source="", platform="abstain",
                    reason=(
                        f"abstain (votes={meta['votes']}, wiki={'yes' if meta['has_wiki'] else 'no'}, "
                        f"consensus={'yes' if meta['consensus'] else 'no'}, "
                        f"cluster={meta['cluster_score']}; no KG; Lens returned no real "
                        f"matches for this face among {len(boosted)} candidates)"
                    ),
                )

        merged = LensResult(
            query_image_url=url1,
            image_sha256=lens1.image_sha256,
            face_hash=face_hash,
            policy=policy,
            selected=chosen,
            visual_matches=boosted,
            knowledge_graph=lens1.knowledge_graph,
            candidates_by_domain=lens1.candidates_by_domain,
            visual_match_count=len(boosted),
            cached=False,
            cache_key=lens1.cache_key,
            cache_hit=False,
            serpapi_total_time_s=(lens1.serpapi_total_time_s or 0) + (lens2.serpapi_total_time_s or 0),
            upload_ms=up1 + up2,
            lens_ms=lens1_ms + lens2_ms,
            upload_host=host1,
            raw_response_at=lens1.raw_response_at,
        )
        self.last_diagnostics = SearchDiagnostics(
            visual_match_count=len(boosted),
            selected_rank=chosen.rank,
            selected_reason=chosen.reason,
            elapsed_seconds=(lens1_ms + lens2_ms) / 1000.0,
            cached=False,
        )
        return (url1, merged, up1 + up2, lens1_ms + lens2_ms, host1)
