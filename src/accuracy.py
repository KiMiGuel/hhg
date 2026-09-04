"""Face quality and accuracy helpers: confidence checks, quality assessment."""

import cv2
import numpy as np

from rich.console import Console

console = Console()

MIN_FACE_CONFIDENCE = 0.5
SFACE_COSINE_THRESHOLD = 0.363  # OpenCV SFace paper default for same-person match


def validate_face_confidence(confidence: float) -> bool:
    """Check if detection confidence meets the minimum threshold."""
    return confidence >= MIN_FACE_CONFIDENCE


def assess_face_quality(image: np.ndarray, bbox: tuple) -> dict:
    """Assess face crop quality: blur (Laplacian variance), brightness, contrast."""
    x, y, w, h = bbox
    face_roi = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi

    # Blur: higher variance = sharper
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_pass = laplacian_var > 100

    # Brightness: mean pixel value
    mean_brightness = float(gray.mean())
    brightness_pass = 40 < mean_brightness < 220

    # Contrast: std deviation
    contrast = float(gray.std())
    contrast_pass = contrast > 30

    return {
        "blur_score": round(laplacian_var, 1),
        "blur_pass": blur_pass,
        "brightness": round(mean_brightness, 1),
        "brightness_pass": brightness_pass,
        "contrast": round(contrast, 1),
        "contrast_pass": contrast_pass,
        "pass": blur_pass and brightness_pass and contrast_pass,
    }


def show_quality_report(quality: dict):
    """Display a quality report to the console."""
    status = "[green]PASS[/green]" if quality["pass"] else "[yellow]MARGINAL[/yellow]"
    console.print(f"\n  [dim]Face quality: {status}[/dim]")
    console.print(f"    Blur score: {quality['blur_score']} {'[green]OK[/green]' if quality['blur_pass'] else '[yellow]LOW[/yellow]'}")
    console.print(f"    Brightness: {quality['brightness']} {'[green]OK[/green]' if quality['brightness_pass'] else '[yellow]LOW[/yellow]'}")
    console.print(f"    Contrast:   {quality['contrast']} {'[green]OK[/green]' if quality['contrast_pass'] else '[yellow]LOW[/yellow]'}")


def normalize_embedding(raw_embedding) -> np.ndarray:
    """L2-normalize a raw SFace embedding to a unit vector."""
    emb = np.array(raw_embedding, dtype=np.float32).flatten()
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized embeddings."""
    a = np.array(emb_a, dtype=np.float32).flatten()
    b = np.array(emb_b, dtype=np.float32).flatten()
    return float(np.dot(a, b))


def same_person(emb_a: np.ndarray, emb_b: np.ndarray) -> bool:
    """Determine if two embeddings represent the same person."""
    return cosine_similarity(emb_a, emb_b) >= SFACE_COSINE_THRESHOLD
