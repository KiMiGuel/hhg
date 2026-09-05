"""Face quality and accuracy helpers: confidence checks, quality assessment.

These gates are deliberately conservative — the cost of *missing* a face in
Hackathon judging is much higher than the cost of *trying* to match a slightly
blurry face, so we warn but don't reject on marginal inputs.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from rich.console import Console

console = Console()

# YuNet's internal score is a probability (0..1) that a detection is a real
# face. We pass `score_threshold=0.3` to FaceDetectorYN.create() so the model
# itself does a coarse filter, then keep detections whose face crops clear
# these independent quality gates below.
MIN_FACE_CONFIDENCE = 0.3

# Default YuNet / SFace paper value for cosine-similarity same-person match.
SFACE_COSINE_THRESHOLD = 0.363

# Minimum face size (max(w, h) in pixels). Below this the embedding is too
# noisy to be reliable. Empirically: faces <60 px are background texture.
MIN_FACE_PIXELS = 60

# Quality gates. These are deliberately permissive — they WARN in the
# report but never fail Stage 1, because for the hackathon demo we want
# the pipeline to run end-to-end on every reasonable input.
BLUR_MIN = 60.0          # Laplacian variance; <60 = very blurry
BRIGHTNESS_MIN = 30.0    # mean gray; <30 = backlit
BRIGHTNESS_MAX = 225.0   # mean gray; >225 = blown out
CONTRAST_MIN = 25.0      # std gray; <25 = flat


def validate_face_confidence(confidence: float) -> bool:
    """Return True if detection confidence meets the minimum threshold."""
    return confidence >= MIN_FACE_CONFIDENCE


def assess_face_quality(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> dict:
    """Assess face crop quality: blur (Laplacian variance), brightness, contrast.

    Returns a dict with `pass`=True only if all three sub-tests pass.
    Used to WARN the user about a bad input but not to reject it.
    """
    x, y, w, h = bbox
    face_roi = image[y:y + h, x:x + w]
    if face_roi.size == 0:
        return _fail_dict("empty crop")
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_pass = laplacian_var > BLUR_MIN

    mean_brightness = float(gray.mean())
    brightness_pass = BRIGHTNESS_MIN < mean_brightness < BRIGHTNESS_MAX

    contrast = float(gray.std())
    contrast_pass = contrast > CONTRAST_MIN

    return {
        "blur_score": round(laplacian_var, 1),
        "blur_pass": blur_pass,
        "brightness": round(mean_brightness, 1),
        "brightness_pass": brightness_pass,
        "contrast": round(contrast, 1),
        "contrast_pass": contrast_pass,
        "pass": blur_pass and brightness_pass and contrast_pass,
    }


def _fail_dict(reason: str) -> dict:
    return {
        "blur_score": 0.0, "blur_pass": False,
        "brightness": 0.0, "brightness_pass": False,
        "contrast": 0.0, "contrast_pass": False,
        "pass": False,
        "reason": reason,
    }


def show_quality_report(quality: dict) -> None:
    """Display a quality report to the console."""
    status = "[green]PASS[/green]" if quality["pass"] else "[yellow]MARGINAL[/yellow]"
    console.print(f"\n  [dim]Face quality: {status}[/dim]")
    console.print(
        f"    Blur score: {quality['blur_score']} "
        f"{'[green]OK[/green]' if quality['blur_pass'] else '[yellow]LOW[/yellow]'}"
    )
    console.print(
        f"    Brightness: {quality['brightness']} "
        f"{'[green]OK[/green]' if quality['brightness_pass'] else '[yellow]LOW[/yellow]'}"
    )
    console.print(
        f"    Contrast:   {quality['contrast']} "
        f"{'[green]OK[/green]' if quality['contrast_pass'] else '[yellow]LOW[/yellow]'}"
    )


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


def l2_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Euclidean L2 distance between two normalized embeddings (range 0..2)."""
    a = np.array(emb_a, dtype=np.float32).flatten()
    b = np.array(emb_b, dtype=np.float32).flatten()
    return float(np.linalg.norm(a - b))


def detection_quality_score(confidence: float, quality: dict) -> float:
    """Composite 0..100 quality score for the detection. Used in the report
    so the auditor can see how trustworthy the biometric hash is.

    Components:
      - 40% confidence (YuNet internal score)
      - 20% blur (Laplacian / 200, capped at 1)
      - 20% brightness (1 - distance from ideal 128 / 128)
      - 20% contrast (std / 80, capped at 1)
    """
    c = max(0.0, min(1.0, confidence))
    b = min(1.0, float(quality.get("blur_score", 0)) / 200.0)
    br = float(quality.get("brightness", 128))
    bright = max(0.0, 1.0 - abs(br - 128.0) / 128.0)
    ct = min(1.0, float(quality.get("contrast", 0)) / 80.0)
    return round(100.0 * (0.40 * c + 0.20 * b + 0.20 * bright + 0.20 * ct), 1)