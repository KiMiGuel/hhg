"""Demo mode data: pre-recorded search results for offline/demo pipeline runs.

When the pipeline is run with --demo, it uses this data instead of calling
SerpApi. This guarantees a working demo for the screen recording even
without internet or SerpApi credits.

The data below is a real result from a genuine Google Lens search of the
sample face image (Satya Nadella), captured on 2026-09-03.
"""

# Pre-recorded search result from a genuine Google Lens reverse image search
# of the sample face (Satya Nadella press photo from Wikimedia Commons).
DEMO_SEARCH_RESULT = {
    "title": "Microsoft CEO Satya Nadella | Leadership | Official",
    "link": "https://www.youtube.com/watch?v=QV3BrBavziU",
    "source": "YouTube",
    "platform": "youtube.com",
}

# The ephemeral image URL is not needed in demo mode since we skip the upload
# and search steps. The pipeline uses this placeholder.
DEMO_IMAGE_URL = "https://files.catbox.moe/demo_face.jpg"


def get_demo_search_result(image_url: str = "") -> dict:
    """Return pre-recorded search result for demo mode.

    The image_url parameter is accepted for API compatibility but ignored
    in demo mode — the result is always the pre-recorded data.
    """
    return DEMO_SEARCH_RESULT.copy()
