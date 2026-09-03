import json
import os
import hashlib
import requests
from serpapi import GoogleSearch

# Cache directory for SerpApi results (avoids consuming credits on re-runs)
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Dynamic whitelist of social platforms for result filtering
SOCIAL_PLATFORMS = [
    "instagram.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "facebook.com",
    "reddit.com",
    "youtube.com",
    "pinterest.com",
]

# Ephemeral / zero-auth public image hosts (primary first, fallback second).
# The primary host must be directly crawlable by Google's image fetcher for
# Google Lens to ingest the photo.
CATBOX_UPLOAD_URL = "https://catbox.moe/user/api.php"
TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"


def filter_social_match(visual_matches: list) -> dict | None:
    """Pure helper: scan Lens visual matches against the social platform
    whitelist and return the first genuine social post, or None."""
    for match in visual_matches or []:
        link = match.get("link", "") or ""
        for platform in SOCIAL_PLATFORMS:
            if platform in link:
                return {
                    "title": match.get("title", "Social Profile Match"),
                    "link": link,
                    "source": match.get("source", platform),
                    "platform": platform,
                }
    return None


class WebSearchEngine:
    """Stage 2: ephemeral image hosting + genuine Google Lens reverse search."""

    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key

    def _cache_key(self, image_url: str) -> str:
        """Generate a deterministic cache key for an image URL."""
        return hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16]

    def _get_cached_result(self, image_url: str) -> dict | None:
        """Return cached search result if it exists, else None."""
        key = self._cache_key(image_url)
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _cache_result(self, image_url: str, result: dict):
        """Save a search result to the cache."""
        key = self._cache_key(image_url)
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except OSError:
            pass  # caching is best-effort

    def upload_image_to_temp_host(self, file_path: str) -> str:
        """Upload the cropped face to a free zero-auth public host and return a
        direct image URL that Google Lens can crawl.

        Primary: catbox.moe (permanent direct links, no account).
        Fallback: tmpfiles.org (ephemeral links).
        """
        # --- Primary: catbox.moe ---
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(
                    CATBOX_UPLOAD_URL,
                    data={"reqtype": "fileupload", "userhash": ""},
                    files={"fileToUpload": f},
                    timeout=60,
                )
            if resp.status_code == 200 and resp.text.startswith("https://"):
                return resp.text.strip()
        except requests.RequestException:
            pass  # fall through to tmpfiles

        # --- Fallback: tmpfiles.org ---
        with open(file_path, "rb") as f:
            resp = requests.post(TMPFILES_UPLOAD_URL, files={"file": f}, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Image hosting failed on both hosts (tmpfiles HTTP {resp.status_code})."
            )
        raw_url = resp.json()["data"]["url"]
        # Convert the preview URL to the direct download URL: tmpfiles.org/xyz -> tmpfiles.org/dl/xyz
        return raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")

    def search_face_on_web(self, image_url: str) -> dict:
        """Run a genuine Google Lens search via SerpApi and dynamically filter
        results against the social media whitelist."""
        if not self.serpapi_key:
            raise ValueError(
                "SERPAPI_KEY is not set. Get a free key (no credit card) at https://serpapi.com"
            )

        # Check cache first to avoid consuming SerpApi credits
        cached = self._get_cached_result(image_url)
        if cached:
            return cached

        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": self.serpapi_key,
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        visual_matches = results.get("visual_matches", [])

        if not visual_matches:
            # Fallback to the knowledge graph if Lens returns no visual matches
            knowledge_graph = results.get("knowledge_graph", [])
            if knowledge_graph and "link" in knowledge_graph[0]:
                result = {
                    "title": knowledge_graph[0].get("title", "Discovered Profile"),
                    "link": knowledge_graph[0].get("link"),
                    "source": "Web Knowledge Graph",
                    "platform": "Web Profile",
                }
                self._cache_result(image_url, result)
                return result
            raise RuntimeError(
                "No visual matches found on the web for the provided face scan.\n"
                "  This can happen if:\n"
                "  - The face has no public web presence (try a well-known public figure)\n"
                "  - The image host URL is not crawlable by Google (try a different host)\n"
                "  - SerpApi returned an error (check your API key and quota)"
            )

        # Prefer the first genuine social media post (pure, testable filter)
        social = filter_social_match(visual_matches)
        if social:
            self._cache_result(image_url, social)
            return social

        # Fallback to the top visual match if no social domain matched exactly
        top_match = visual_matches[0]
        result = {
            "title": top_match.get("title", "Online Entity Match"),
            "link": top_match.get("link", ""),
            "source": top_match.get("source", "Web"),
            "platform": "Discovered Web Article/Profile",
        }
        self._cache_result(image_url, result)
        return result
