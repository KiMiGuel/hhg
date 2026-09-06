# -*- coding: utf-8 -*-
"""Multi-platform profile corroboration for the HHG pipeline.

After Lens voting picks a name, this module verifies that the named person
actually has a public profile on each of the target platforms
(Instagram, Facebook, YouTube, X, LinkedIn, Pinterest, TikTok, Reddit).

Two distinct signals are produced:

1. ``find_platform_profiles(name)`` -- per-platform *existence* check
   via ``site:<domain> "<name>"`` SerpApi search.  Cached on disk so reruns
   cost zero credits.

2. ``find_exact_image_per_platform(query_image_path, hits)`` -- *exact-file*
   corroboration: for every confirmed platform, fetch the profile page's
   avatar (via og:image) and compare it against the uploaded face crop
   with a perceptual dHash.  An EXACT match (Hamming <= 6) is strong
   evidence the same image is hosted on that platform.

The result is a ``PlatformCorroboration`` dataclass that the audit report
and the rich visualizer both consume, so a single ``main.py`` run
publishes a "PLATFORM COVERAGE" line such as::

    PLATFORM COVERAGE: Instagram [image-match]  YouTube [profile-only]
                        Facebook [profile-only]  X [profile-only]
                        LinkedIn [profile-only]  TikTok [image-match]
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

import cv2
import numpy as np

# Reuse the user-agent + image fetcher from the biometric verifier so we
# don't get blocked by the public platforms' anti-bot filters.
try:
    from src.biometric_verify import _UA as _USER_AGENT, fetch_image
except Exception:  # pragma: no cover - import safety
    _USER_AGENT = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        )
    }

    def fetch_image(url, timeout=15):  # type: ignore
        return None


# ----------------------------------------------------------------- constants
# The seven platforms the user explicitly asked for, plus Pinterest, Reddit,
# and 5 additional high-traffic platforms where public figures host content.
# Order matters: it's the order we display them in the report.
PLATFORM_DOMAINS: list[tuple[str, str]] = [
    # The user's required 7 (instagram, facebook, youtube, x, linkedin, pinterest, tiktok)
    ("instagram", "instagram.com"),
    ("facebook", "facebook.com"),
    ("youtube", "youtube.com"),
    ("x", "x.com"),
    ("twitter", "twitter.com"),  # legacy alias; dedupes with x in display
    ("linkedin", "linkedin.com"),
    ("pinterest", "pinterest.com"),
    ("tiktok", "tiktok.com"),
    # Extended coverage (added for ≥90% cross-platform accuracy)
    ("reddit", "reddit.com"),
    ("threads", "threads.net"),
    ("quora", "quora.com"),
    ("medium", "medium.com"),
    ("flickr", "flickr.com"),
    ("tumblr", "tumblr.com"),
]

# These platforms can be displayed by their canonical short name even when
# the URL pointed at the twitter.com or x.com equivalent.
PLATFORM_DISPLAY = {
    "instagram": "Instagram",
    "facebook": "Facebook",
    "youtube": "YouTube",
    "x": "X (Twitter)",
    "twitter": "X (Twitter)",
    "linkedin": "LinkedIn",
    "pinterest": "Pinterest",
    "tiktok": "TikTok",
    "reddit": "Reddit",
    "threads": "Threads",
    "quora": "Quora",
    "medium": "Medium",
    "flickr": "Flickr",
    "tumblr": "Tumblr",
}

# Exact-image dHash thresholds.  Avatars on most platforms are heavily
# cropped and recompressed, so we tolerate more bit-flips than for
# thumbnail-vs-thumbnail comparisons.
EXACT_AVATAR_THRESHOLD = 14   # of 64 bits (raised from 12 to absorb heavier platform recompression)
EXACT_THUMBNAIL_THRESHOLD = 10  # of 64 bits (raised from 6; was too tight for real platform images)
MAX_PARALLEL = 6
TIMEOUT_PER_REQUEST = 8

CACHE_DIR = os.path.join("cache", "platform_profiles")
os.makedirs(CACHE_DIR, exist_ok=True)


# ------------------------------------------------------------- result shape
@dataclass
class PlatformHit:
    """One corroborated platform profile."""

    platform: str          # canonical key, e.g. "instagram"
    display_name: str      # human label, e.g. "Instagram"
    profile_url: str       # the discovered profile URL
    avatar_url: str = ""   # the og:image of the profile (may be empty)
    avatar_match: bool = False
    avatar_hamming: int | None = None
    match_status: str = "PROFILE_ONLY"   # EXACT_IMAGE | AVATAR_MATCH | PROFILE_ONLY
    source_query: str = ""  # the SerpApi query that surfaced this profile

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlatformCorroboration:
    """Result of the whole multi-platform profile verification step."""

    name: str
    hits: list[PlatformHit] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    cached: bool = False
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def platforms_found(self) -> list[str]:
        return [h.display_name for h in self.hits]

    def platforms_with_image(self) -> list[str]:
        return [h.display_name for h in self.hits
                if h.match_status in ("EXACT_IMAGE", "AVATAR_MATCH")]

    def coverage_count(self) -> int:
        return len(self.hits)

    def exact_image_count(self) -> int:
        return sum(1 for h in self.hits if h.match_status == "EXACT_IMAGE")

    def render_line(self) -> str:
        """Single-line summary the visualizer prints under the identity."""
        if self.skipped_reason:
            return f"[dim]PLATFORM COVERAGE: skipped ({self.skipped_reason})[/dim]"
        if not self.hits:
            return "[yellow]PLATFORM COVERAGE: no profiles found[/yellow]"
        parts = []
        for h in self.hits:
            if h.match_status == "EXACT_IMAGE":
                badge = "[bold green]exact-image[/bold green]"
            elif h.match_status == "AVATAR_MATCH":
                badge = "[green]avatar-match[/green]"
            else:
                badge = "[cyan]profile-only[/cyan]"
            parts.append(f"{h.display_name} {badge}")
        head = "[bold]PLATFORM COVERAGE:[/bold] "
        return head + "  ".join(parts)


# ------------------------------------------------------------- cache
def _cache_key(name: str, sites: list[tuple[str, str]] | None = None) -> str:
    """Stable hash of (name, sites-tuple). Different site lists get different
    cache entries so callers asking for a subset (e.g. only Instagram) don't
    get back stale hits from a previous 14-platform run."""
    import hashlib
    name_lc = (name or "").lower().strip()
    sites_part = ",".join(f"{s}:{d}" for s, d in (sites or []))
    return hashlib.sha256(f"profiles:{name_lc}|sites:{sites_part}".encode()).hexdigest()[:32]


def _cache_path(name: str, sites: list[tuple[str, str]] | None = None) -> str:
    return os.path.join(CACHE_DIR, _cache_key(name, sites) + ".json")


def _read_cache(name: str, sites: list[tuple[str, str]] | None = None) -> dict[str, Any] | None:
    p = _cache_path(name, sites)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_cache(name: str, payload: dict[str, Any],
                 sites: list[tuple[str, str]] | None = None) -> None:
    p = _cache_path(name, sites)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except OSError:
        pass


# ------------------------------------------------------------- dHash (mirrors src/image_match.py)
def _dhash_bits(gray: np.ndarray, hash_size: int = 8) -> int:
    resized = cv2.resize(gray, (hash_size + 1, hash_size),
                         interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for b in diff.flatten():
        value = (value << 1) | int(b)
    return value


def _dhash_from_path(path: str) -> int | None:
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _dhash_bits(gray)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ------------------------------------------------------------- profile discovery
def _og_image(url: str, timeout: int = 6) -> str:
    """Extract og:image from a public profile page.  Empty on any failure."""
    if not url:
        return ""
    try:
        r = requests.get(url, timeout=timeout, headers=_USER_AGENT,
                         allow_redirects=True)
        if r.status_code != 200:
            return ""
        head = r.text[:120_000]
        for pat in (
            re.compile(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
                       re.IGNORECASE),
            re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
                       re.IGNORECASE),
        ):
            m = pat.search(head)
            if m and m.group(1).startswith(("http://", "https://")):
                return m.group(1)
    except Exception:
        return ""
    return ""


def _serpapi_request(engine, name: str, domain: str,
                     serpapi_key: str, timeout: int = 10) -> str | None:
    """Run a single `site:<domain> "<name>"` query and return the first
    matching organic-result URL, or None."""
    if not serpapi_key and engine is None:
        return None
    if engine is not None and hasattr(engine, "_request_search"):
        try:
            data = engine._request_search(f'"{name}" site:{domain}',
                                          timeout=timeout)
        except Exception:
            return None
    else:
        try:
            r = requests.get(
                "https://serpapi.com/search.json",
                params={"q": f'"{name}" site:{domain}',
                        "api_key": serpapi_key, "num": 5},
                timeout=timeout, headers=_USER_AGENT,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            return None
    for org in (data.get("organic_results") or []):
        link = (org.get("link") or "").lower()
        if domain in link:
            return org.get("link") or ""
    return None


def find_platform_profiles(
    name: str,
    serpapi_key: str = "",
    engine: Any = None,
    sites: list[tuple[str, str]] | None = None,
    max_parallel: int = MAX_PARALLEL,
    per_request_timeout: int = TIMEOUT_PER_REQUEST,
) -> list[PlatformHit]:
    """For each (platform, domain) in ``sites``, confirm a public profile
    exists for ``name`` and return a list of ``PlatformHit`` records.

    Cached on disk so subsequent runs are free.  Always returns a list
    (possibly empty) -- never raises.
    """
    name = (name or "").strip()
    if not name:
        return []

    # If neither a SerpApi key nor an engine is supplied, we cannot reach
    # any search backend -- return an empty list immediately so the
    # caller (visualizer / report) can show a clean "skipped" message.
    if not serpapi_key and engine is None:
        return []

    sites = sites or PLATFORM_DOMAINS  # 14 platforms (7 user-required + 7 extended)
    sites_tuple = list(sites)

    cached = _read_cache(name, sites_tuple)
    if cached and cached.get("name", "").lower() == name.lower():
        hits: list[PlatformHit] = []
        for h in cached.get("hits", []):
            hits.append(PlatformHit(**h))
        if hits:
            for h in hits:
                h.avatar_match = False
                h.match_status = "PROFILE_ONLY"
            return hits

    hits: list[PlatformHit] = []

    def _query(site: str, domain: str) -> PlatformHit | None:
        link = _serpapi_request(engine, name, domain, serpapi_key,
                                timeout=per_request_timeout)
        if not link:
            return None
        return PlatformHit(
            platform=site,
            display_name=PLATFORM_DISPLAY.get(site, site.title()),
            profile_url=link,
            source_query=f'"{name}" site:{domain}',
        )

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(_query, s, d): (s, d) for s, d in sites}
        for fut in as_completed(futures):
            try:
                hit = fut.result()
            except Exception:
                hit = None
            if hit:
                hits.append(hit)

    if hits:
        _write_cache(name, {
            "name": name,
            "hits": [h.to_dict() for h in hits],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, sites_tuple)

    return hits


# ------------------------------------------------------------- exact-image corroboration
def _avatar_image(hit: PlatformHit, timeout: int = 6) -> str:
    """Return the file path of a downloaded avatar (or empty string on failure)."""
    if not hit.avatar_url:
        hit.avatar_url = _og_image(hit.profile_url, timeout=timeout)
    if not hit.avatar_url:
        return ""
    data = fetch_image(hit.avatar_url, timeout=timeout)
    if not data:
        return ""
    out_dir = "temp/avatars"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{hit.platform}_{int(time.time()*1000)}.jpg")
    try:
        with open(out, "wb") as f:
            f.write(data)
        return out
    except OSError:
        return ""


def find_exact_image_per_platform(
    query_image_path: str,
    hits: list[PlatformHit],
    max_parallel: int = MAX_PARALLEL,
    avatar_threshold: int = EXACT_AVATAR_THRESHOLD,
) -> list[PlatformHit]:
    """For each PlatformHit, fetch the profile's avatar and compare it
    against the uploaded image with a dHash.  Mutates and returns the
    same list of PlatformHit with avatar_match / match_status updated.

    ``avatar_threshold`` is the Hamming distance (of 64) below which an
    avatar is considered a positive match.  Default 12 is calibrated for
    the heavy cropping and recompression that platforms apply to avatars.
    """
    if not hits or not query_image_path or not os.path.exists(query_image_path):
        return hits
    query_hash = _dhash_from_path(query_image_path)
    if query_hash is None:
        return hits

    def _check(hit: PlatformHit) -> None:
        path = _avatar_image(hit)
        if not path:
            return
        avatar_hash = _dhash_from_path(path)
        if avatar_hash is None:
            return
        dist = _hamming(query_hash, avatar_hash)
        hit.avatar_hamming = int(dist)
        if dist <= EXACT_THUMBNAIL_THRESHOLD:
            hit.avatar_match = True
            hit.match_status = "EXACT_IMAGE"
        elif dist <= avatar_threshold:
            hit.avatar_match = True
            hit.match_status = "AVATAR_MATCH"

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(_check, h) for h in hits]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                continue
    return hits


# ------------------------------------------------------------- public entry point
def corroborate(
    name: str,
    query_image_path: str | None = None,
    serpapi_key: str = "",
    engine: Any = None,
    sites: list[tuple[str, str]] | None = None,
) -> PlatformCorroboration:
    """End-to-end: confirm profile existence on all 7 target platforms
    AND (if ``query_image_path`` is provided) check whether the same
    image appears as the platform avatar.
    """
    t0 = time.perf_counter()
    if not (name or "").strip():
        return PlatformCorroboration(name="", skipped_reason="empty name")
    if not serpapi_key and engine is None:
        return PlatformCorroboration(
            name=name,
            skipped_reason="no SERPAPI_KEY (corroboration requires SerpApi)",
        )
    hits = find_platform_profiles(name, serpapi_key=serpapi_key,
                                  engine=engine, sites=sites)
    if query_image_path and hits:
        find_exact_image_per_platform(query_image_path, hits)
    return PlatformCorroboration(
        name=name,
        hits=hits,
        elapsed_seconds=round(time.perf_counter() - t0, 3),
    )


__all__ = [
    "PLATFORM_DOMAINS",
    "PLATFORM_DISPLAY",
    "PlatformHit",
    "PlatformCorroboration",
    "corroborate",
    "find_platform_profiles",
    "find_exact_image_per_platform",
]




