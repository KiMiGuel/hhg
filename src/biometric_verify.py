"""Biometric re-verification of the selected identity.

After the text-based selection (Lens voting) picks a candidate, this module
fetches the candidate's profile photo (Wikipedia portrait via the public REST
API, or og:image from the candidate/profile page) and compares it against the
query face embedding with SFace cosine similarity — turning text-only
selection into actual face verification.

All network helpers degrade gracefully: any failure means "unknown", never a
false verdict.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

import numpy as np
import requests

from src.face_engine import SFACE_COSINE_THRESHOLD

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

_OG_IMAGE_PATTERNS = [
    re.compile(
        r"<meta[^>]+property=[\"']og:image[\"'][^>]+content=[\"']([^\"']+)[\"']", re.IGNORECASE
    ),
    re.compile(
        r"<meta[^>]+content=[\"']([^\"']+)[\"'][^>]+property=[\"']og:image[\"']", re.IGNORECASE
    ),
]

# Cosine bands for the biometric verdict. SFace's reference threshold is
# 0.363; >=0.45 is a strong match in practice on web profile photos.
BIOMETRIC_HIGH = 0.45
BIOMETRIC_MEDIUM = SFACE_COSINE_THRESHOLD  # 0.363


def fetch_image(url: str, timeout: int = 15) -> bytes | None:
    """Download image bytes; None on any failure or non-image payload."""
    if not url:
        return None
    try:
        r = requests.get(url, timeout=timeout, headers=_UA, allow_redirects=True)
        is_img = r.status_code == 200 and (
            "image" in r.headers.get("content-type", "")
            or r.content[:3] in (b"\xff\xd8\xff", b"\x89PNG")
        )
        return r.content if is_img else None
    except Exception:
        return None


def _og_image(url: str, timeout: int = 10) -> str | None:
    """Extract og:image from a page's HTML (public profiles only)."""
    if not url:
        return None
    try:
        r = requests.get(url, timeout=timeout, headers=_UA, allow_redirects=True, stream=False)
        if r.status_code != 200:
            return None
        head = r.text[:250_000]
        for pat in _OG_IMAGE_PATTERNS:
            m = pat.search(head)
            if m and m.group(1).startswith("http"):
                return m.group(1)
    except Exception:
        pass
    return None


def _wikipedia_photo(link: str) -> str | None:
    """High-res portrait for a Wikipedia person page (public REST API)."""
    if not link:
        return None
    try:
        p = urlparse(link)
        if "wikipedia.org" not in p.netloc.lower() or "/wiki/" not in p.path:
            return None
        title = unquote(p.path).rsplit("/", 1)[-1]
        api = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        r = requests.get(api, timeout=10, headers=_UA)
        if r.status_code != 200:
            return None
        j = r.json()
        return (j.get("originalimage") or {}).get("source") or (j.get("thumbnail") or {}).get(
            "source"
        )
    except Exception:
        return None


def candidate_photo_urls(selected_link: str, platform_profiles: dict | None = None) -> list[str]:
    """Ordered list of candidate photo URLs for biometric verification.

    Priority: Wikipedia portrait (most reliable, no scraping blocks) ->
    og:image of the selected page -> og:image of corroborated social profiles
    (YouTube channel avatar, Instagram/Facebook profile images).
    """
    urls: list[str] = []
    if selected_link:
        wiki = _wikipedia_photo(selected_link)
        if wiki:
            urls.append(wiki)
        if "wikipedia.org" not in (selected_link or "").lower():
            og = _og_image(selected_link)
            if og:
                urls.append(og)
    for plat in ("youtube", "instagram", "facebook"):
        u = (platform_profiles or {}).get(plat)
        if u:
            og = _og_image(u)
            if og:
                urls.append(og)
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


class BiometricVerifier:
    """Compare the query face embedding against candidate profile photos."""

    HIGH = BIOMETRIC_HIGH
    MEDIUM = BIOMETRIC_MEDIUM
    MAX_PHOTOS = 8  # raised from 4 for better hard-case accuracy
    PARALLEL_FETCHES = 4  # concurrent image fetches

    def __init__(self, face_engine) -> None:
        self.engine = face_engine

    def verify(self, query_embedding, photo_urls: list[str]) -> dict:
        """Return {biometric_confidence, similarity, photo_url, checked}.

        biometric_confidence: HIGH / MEDIUM / LOW / UNKNOWN (no photo could
        be fetched or no face found — never a false verdict).

        Photos are fetched in parallel so wall time ~= single fetch rather
        than sum of fetches, while still respecting MAX_PHOTOS.
        """
        result = {
            "biometric_confidence": "UNKNOWN",
            "similarity": None,
            "photo_url": None,
            "checked": 0,
        }
        if query_embedding is None:
            return result
        q = np.asarray(query_embedding, dtype=np.float32).flatten()
        urls = list((photo_urls or [])[: self.MAX_PHOTOS])
        if not urls:
            return result

        # ---- parallel fetch + embed ----
        from concurrent.futures import ThreadPoolExecutor

        def _check(url: str) -> tuple[str, float | None]:
            data = fetch_image(url)
            if not data:
                return (url, None)
            emb = self.engine.embed_from_bytes(data)
            if emb is None:
                return (url, None)
            return (url, float(self.engine.cosine_similarity(q, emb)))

        best_sim: float | None = None
        best_url: str | None = None
        checked = 0
        with ThreadPoolExecutor(max_workers=self.PARALLEL_FETCHES) as pool:
            for url, sim in pool.map(_check, urls):
                if sim is None:
                    continue
                checked += 1
                if best_sim is None or sim > best_sim:
                    best_sim = sim
                    best_url = url
        result["checked"] = checked
        if best_sim is not None:
            result["similarity"] = round(best_sim, 4)
            result["photo_url"] = best_url
        if best_sim is not None:
            if best_sim >= self.HIGH:
                result["biometric_confidence"] = "HIGH"
            elif best_sim >= self.MEDIUM:
                result["biometric_confidence"] = "MEDIUM"
            else:
                result["biometric_confidence"] = "LOW"
        return result
