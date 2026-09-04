"""Demo data for offline pipeline runs (no SerpApi/internet needed)."""


def get_demo_search_result() -> dict:
    """Return a pre-recorded search result for demo mode."""
    return {
        "title": "Satya Nadella — Microsoft CEO (Demo Result)",
        "link": "https://www.youtube.com/watch?v=demo_placeholder",
        "source": "youtube.com",
        "platform": "youtube.com",
        "snippet": "This is a pre-recorded demo result. In live mode, this would be a genuine Google Lens match.",
        "is_demo": True,
    }
