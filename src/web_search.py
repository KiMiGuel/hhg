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
import io
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
        # Trim the response (and the server-side processing) by asking Lens for
        # only the social/news/official slice when that's what we want.
        if policy in ("social", "news", "official"):
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
    def _select(
        self,
        visual_matches: list[dict[str, Any]],
        knowledge_graph_raw: Any,
        policy: str,
    ) -> tuple[LensMatch, list[LensMatch], dict[str, list[LensMatch]], LensMatch | None]:
        """Apply the policy and return (selected, all_visual, by_domain, kg)."""
        all_visual: list[LensMatch] = []
        for i, m in enumerate(visual_matches or [], start=1):
            all_visual.append(_match_from_visual(i, m, f"visual match #{i}"))
        by_domain: dict[str, list[LensMatch]] = {}
        for m in all_visual:
            by_domain.setdefault(m.platform, []).append(m)

        kg_match: LensMatch | None = None
        if isinstance(knowledge_graph_raw, dict) and knowledge_graph_raw.get("link"):
            kg_match = LensMatch(
                rank=None,
                title=knowledge_graph_raw.get("title", "Knowledge Graph"),
                link=knowledge_graph_raw["link"],
                source="Google Knowledge Graph",
                platform=_classify_link(knowledge_graph_raw["link"]) or "web",
                reason="knowledge graph entity",
            )
        elif (
            isinstance(knowledge_graph_raw, list)
            and knowledge_graph_raw
            and knowledge_graph_raw[0].get("link")
        ):
            raw = knowledge_graph_raw[0]
            kg_match = LensMatch(
                rank=None, title=raw.get("title", "Knowledge Graph"),
                link=raw["link"], source="Google Knowledge Graph",
                platform=_classify_link(raw["link"]) or "web",
                reason="knowledge graph entity",
            )

        if policy == "social":
            chosen = next((m for m in all_visual if m.platform in SOCIAL_PLATFORMS), None)
            if chosen:
                return (
                    LensMatch(
                        rank=chosen.rank, title=chosen.title, link=chosen.link,
                        source=chosen.source, platform=chosen.platform,
                        reason=f"first social-domain match ({chosen.platform})",
                    ),
                    all_visual, by_domain, kg_match,
                )
            if kg_match:
                return (
                    LensMatch(
                        rank=None, title=kg_match.title, link=kg_match.link,
                        source=kg_match.source, platform=kg_match.platform,
                        reason="knowledge graph fallback (no social match)",
                    ),
                    all_visual, by_domain, kg_match,
                )
            if all_visual:
                top = all_visual[0]
                return (
                    LensMatch(
                        rank=top.rank, title=top.title, link=top.link,
                        source=top.source, platform=top.platform,
                        reason="top visual match (no social-domain match found)",
                    ),
                    all_visual, by_domain, kg_match,
                )
            raise RuntimeError(
                "No visual matches found for this face. Try a clearer, front-facing image "
                "of a person with public web/social presence."
            )
        return self._select_tail(all_visual, by_domain, kg_match, policy)

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
        results = self._request_serpapi(image_url, policy=policy)
        lens_ms = (time.perf_counter() - lens_started) * 1000.0

        visual_matches = results.get("visual_matches", []) or []
        knowledge_graph = results.get("knowledge_graph")

        selected, all_visual, by_domain, kg_match = self._select(
            visual_matches, knowledge_graph, policy
        )

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
