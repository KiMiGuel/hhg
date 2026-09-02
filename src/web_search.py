import requests
from serpapi import GoogleSearch

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
                return {
                    "title": knowledge_graph[0].get("title", "Discovered Profile"),
                    "link": knowledge_graph[0].get("link"),
                    "source": "Web Knowledge Graph",
                    "platform": "Web Profile",
                }
            raise RuntimeError("No visual matches found on the web for the provided face scan.")

        # Prefer the first genuine social media post (pure, testable filter)
        social = filter_social_match(visual_matches)
        if social:
            return social

        # Fallback to the top visual match if no social domain matched exactly
        top_match = visual_matches[0]
        return {
            "title": top_match.get("title", "Online Entity Match"),
            "link": top_match.get("link", ""),
            "source": top_match.get("source", "Web"),
            "platform": "Discovered Web Article/Profile",
        }
