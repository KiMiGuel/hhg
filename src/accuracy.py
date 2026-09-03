"""Accuracy validation: face quality assessment and search result verification.

Ensures the pipeline produces correct results, not just fast results.
"""

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel

console = Console()

# Quality thresholds
MIN_FACE_CONFIDENCE = 0.9      # Reject detections below this
MIN_BLUR_VARIANCE = 100.0      # Laplacian variance (lower = blurrier)
MIN_BRIGHTNESS = 40.0          # Mean pixel value (0-255)
MAX_BRIGHTNESS = 250.0         # Avoid overexposed
MIN_FACE_SIZE = 64             # Minimum face dimension in pixels


def assess_face_quality(image: np.ndarray, bbox: tuple) -> dict:
    """Assess the quality of a detected face for reliable embedding extraction.

    Returns a quality report with pass/fail status and actionable warnings.
    """
    x, y, w, h = bbox
    face_roi = image[y:y+h, x:x+w]

    if face_roi.size == 0:
        return {"pass": False, "reason": "Empty face region"}

    # Convert to grayscale for analysis
    if len(face_roi.shape) == 3:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_roi

    # Blur detection via Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness check
    mean_brightness = float(np.mean(gray))

    # Face size check
    min_dim = min(w, h)

    issues = []
    if laplacian_var < MIN_BLUR_VARIANCE:
        issues.append(
            f"Face appears blurry (sharpness: {laplacian_var:.1f}, "
            f"minimum: {MIN_BLUR_VARIANCE})"
        )
    if mean_brightness < MIN_BRIGHTNESS:
        issues.append(
            f"Face too dark (brightness: {mean_brightness:.1f}, "
            f"minimum: {MIN_BRIGHTNESS})"
        )
    if mean_brightness > MAX_BRIGHTNESS:
        issues.append(
            f"Face overexposed (brightness: {mean_brightness:.1f}, "
            f"maximum: {MAX_BRIGHTNESS})"
        )
    if min_dim < MIN_FACE_SIZE:
        issues.append(
            f"Face too small ({min_dim}px, minimum: {MIN_FACE_SIZE}px)"
        )

    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "sharpness": laplacian_var,
        "brightness": mean_brightness,
        "face_size": min_dim,
    }


def validate_face_confidence(confidence: float) -> bool:
    """Check if face detection confidence meets the threshold."""
    return confidence >= MIN_FACE_CONFIDENCE


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize an SFace embedding for consistent cosine similarity.

    Normal embeddings produce more reliable similarity comparisons.
    """
    vec = embedding.flatten().astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return (vec / norm).astype(np.float32)


def validate_search_result(search_result: dict, knowledge_graph: list = None) -> dict:
    """Validate a search result against the knowledge graph.

    Cross-checks the discovered post title with the person's name from
    the knowledge graph. Returns a validation report.
    """
    result = {
        "validated": False,
        "confidence": "low",
        "reason": "",
    }

    if not search_result or not search_result.get("link"):
        result["reason"] = "No search result or link found"
        return result

    # If we have a knowledge graph, cross-check the name
    if knowledge_graph and len(knowledge_graph) > 0:
        kg_title = knowledge_graph[0].get("title", "").lower()
        result_title = search_result.get("title", "").lower()

        # Extract potential name from knowledge graph
        kg_words = set(kg_title.split())
        result_words = set(result_title.split())

        # Check for name overlap
        common_words = kg_words & result_words
        # Filter out common stop words
        stop_words = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or"}
        name_words = common_words - stop_words

        if len(name_words) >= 2:
            result["validated"] = True
            result["confidence"] = "high"
            result["reason"] = f"Name match: {' '.join(name_words)}"
        elif len(name_words) >= 1:
            result["validated"] = True
            result["confidence"] = "medium"
            result["reason"] = f"Partial name match: {' '.join(name_words)}"
        else:
            result["reason"] = "No name match between search result and knowledge graph"
    else:
        # No knowledge graph to validate against — accept result but flag low confidence
        result["validated"] = True
        result["confidence"] = "low"
        result["reason"] = "No knowledge graph available for validation"

    return result


def show_quality_report(quality: dict):
    """Display a face quality report to the user."""
    if quality["pass"]:
        console.print(
            Panel(
                f"[bold green]Face quality check PASSED[/bold green]\n"
                f"  Sharpness: {quality['sharpness']:.1f}\n"
                f"  Brightness: {quality['brightness']:.1f}\n"
                f"  Face size: {quality['face_size']}px",
                border_style="green",
            )
        )
    else:
        issues_text = "\n".join(f"  • {issue}" for issue in quality["issues"])
        console.print(
            Panel(
                f"[bold yellow]Face quality warnings[/bold yellow]\n{issues_text}\n\n"
                f"[dim]Results may be less reliable. Consider using a clearer photo.[/dim]",
                border_style="yellow",
            )
        )
