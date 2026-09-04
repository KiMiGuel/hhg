"""Stage 2: real image hosting + Google Lens search via SerpApi.

This module intentionally performs a genuine live search when called from the
pipeline. It does not contain hardcoded result URLs. It also avoids the SerpApi
SDK's occasionally blocking `get_dict()` call by using `requests.get` with
explicit timeouts and retry/backoff.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

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

CATBOX_UPLOAD_URL = "https://catbox.moe/user/api.php"
TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"


@dataclass(frozen=True)
class SearchDiagnostics:
    """Extra audit data for Stage 2."""

    visual_match_count: int
    selected_rank: int | None
    selected_reason: str
    elapsed_seconds: float
    cached: bool = False


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
    """Ephemeral image hosting + genuine Google Lens reverse search."""

    def __init__(self, serpapi_key: str, timeout: int = 90, retries: int = 2):
        self.serpapi_key = (serpapi_key or "").strip().strip('"').strip("'")
        self.timeout = timeout
        self.retries = retries
        self.last_diagnostics: SearchDiagnostics | None = None

    def _cache_key(self, image_url: str) -> str:
        return hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:24]

    def _cache_path(self, image_url: str) -> str:
        return os.path.join(CACHE_DIR, f"{self._cache_key(image_url)}.json")

    def _get_cached_result(self, image_url: str) -> dict[str, Any] | None:
        path = self._cache_path(image_url)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.last_diagnostics = SearchDiagnostics(
                visual_match_count=int(data.get("visual_match_count", 0)),
                selected_rank=data.get("rank"),
                selected_reason=data.get("reason", "cached result"),
                elapsed_seconds=0.0,
                cached=True,
            )
            return data
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def _cache_result(self, image_url: str, result: dict[str, Any]) -> None:
        try:
            with open(self._cache_path(image_url), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def upload_image_to_temp_host(self, file_path: str) -> str:
        """Upload image and return a direct public URL crawlable by Google Lens.

        Primary: catbox.moe. Fallback: tmpfiles.org.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot upload missing image: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Cannot upload empty image: {file_path}")

        # Catbox is used first because it returns direct crawlable links.
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(
                    CATBOX_UPLOAD_URL,
                    data={"reqtype": "fileupload", "userhash": ""},
                    files={"fileToUpload": f},
                    timeout=45,
                )
            if resp.status_code == 200 and resp.text.strip().startswith("https://"):
                return resp.text.strip()
        except requests.RequestException:
            pass

        # Fallback: tmpfiles.
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(TMPFILES_UPLOAD_URL, files={"file": f}, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"tmpfiles.org returned HTTP {resp.status_code}: {resp.text[:200]}")
            raw_url = resp.json()["data"]["url"]
            return raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        except Exception as e:
            raise RuntimeError(f"Image upload failed on catbox.moe and tmpfiles.org: {e}") from e

    def _request_serpapi(self, image_url: str) -> dict[str, Any]:
        if not self.serpapi_key:
            raise ValueError("SERPAPI_KEY is not set. Get a free key at https://serpapi.com")

        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": self.serpapi_key,
            "no_cache": "false",
        }
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
                    time.sleep(min(2**attempt, 8))
                    continue
                break
        raise RuntimeError(f"SerpApi search failed after {self.retries + 1} attempt(s): {last_error}")

    def search_face_on_web(self, image_url: str) -> dict[str, Any]:
        """Run real Google Lens search and return a selected match.

        Selection policy:
        1. First matching social-media domain from the whitelist.
        2. Knowledge graph link, if no visual matches exist.
        3. Top visual match as transparent fallback.
        """
        cached = self._get_cached_result(image_url)
        if cached:
            return cached

        started = time.perf_counter()
        results = self._request_serpapi(image_url)
        elapsed = time.perf_counter() - started
        visual_matches = results.get("visual_matches", []) or []

        if visual_matches:
            social = filter_social_match(visual_matches)
            if social:
                result = dict(social)
                result["visual_match_count"] = len(visual_matches)
                self.last_diagnostics = SearchDiagnostics(
                    visual_match_count=len(visual_matches),
                    selected_rank=result.get("rank"),
                    selected_reason=result.get("reason", "social match"),
                    elapsed_seconds=elapsed,
                    cached=False,
                )
                self._cache_result(image_url, result)
                return result

            top = visual_matches[0]
            result = {
                "title": top.get("title", "Online Entity Match"),
                "link": top.get("link", ""),
                "source": top.get("source", "Web"),
                "platform": "Discovered Web Article/Profile",
                "rank": 1,
                "reason": "top visual match (no social whitelist domain found)",
                "visual_match_count": len(visual_matches),
            }
            self.last_diagnostics = SearchDiagnostics(
                visual_match_count=len(visual_matches),
                selected_rank=1,
                selected_reason=result["reason"],
                elapsed_seconds=elapsed,
                cached=False,
            )
            self._cache_result(image_url, result)
            return result

        knowledge_graph = results.get("knowledge_graph", []) or []
        if knowledge_graph and isinstance(knowledge_graph, list) and knowledge_graph[0].get("link"):
            kg = knowledge_graph[0]
            result = {
                "title": kg.get("title", "Discovered Profile"),
                "link": kg.get("link"),
                "source": "Web Knowledge Graph",
                "platform": "Web Profile",
                "rank": None,
                "reason": "knowledge graph fallback",
                "visual_match_count": 0,
            }
            self.last_diagnostics = SearchDiagnostics(0, None, result["reason"], elapsed, False)
            self._cache_result(image_url, result)
            return result

        raise RuntimeError(
            "No visual matches found for this face. Try a clearer, front-facing image "
            "of a person with public web/social presence."
        )
