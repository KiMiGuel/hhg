"""Camera capture module: real-time webcam face detection with capture.

Press SPACE to capture the BEST recent frame (with the largest face).
Press ESC to cancel.
Press 'f' to toggle fullscreen preview.
Press 'a' to toggle auto-capture (saves as soon as quality passes).

The capture is quality-gated: a rolling buffer of recent frames is scored by
detection confidence x face size x sharpness, and the BEST frame in the
buffer is saved -- so a momentary blur at the instant you press SPACE no
longer ruins the search. Dim frames are auto-enhanced with CLAHE.
"""

import collections
import os

import cv2
import numpy as np
from rich.console import Console

console = Console()

CAPTURE_DIR = "data"
DEFAULT_OUTPUT = os.path.join(CAPTURE_DIR, "captured_face.jpg")

# Rolling best-shot buffer length (~2s at 30fps). Long enough to ride out a
# blink/blur, short enough that the saved frame is what the user intended.
BUFFER_SECONDS = 2.0
ESTIMATED_FPS = 30
MAX_BUFFER = int(BUFFER_SECONDS * ESTIMATED_FPS)

# Low-light threshold (mean V-channel). Below this, CLAHE is applied.
LOW_LIGHT_MEAN = 90


def _detect_faces_frame(frame, detector, min_confidence=0.5):
    """Run YuNet detection on a single frame. Returns list of (x, y, w, h, score)."""
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    results = []
    for row in faces:
        x, y, fw, fh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        score = float(row[14])
        if score >= min_confidence:
            results.append((x, y, fw, fh, score))
    return results


def _frame_quality(frame, faces) -> dict:
    """Composite capture score for one frame + diagnostics for the overlay.

    score = detection confidence x size factor x sharpness factor.
    """
    if not faces:
        return {"score": 0.0, "brightness": 0.0, "sharpness": 0.0, "conf": 0.0, "face_ratio": 0.0}
    x, y, w, h, conf = max(faces, key=lambda f: f[2] * f[3])
    ih, iw = frame.shape[:2]
    face_ratio = (w * h) / float(iw * ih)
    roi = frame[max(0, y) : y + h, max(0, x) : x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.size else np.zeros((8, 8), np.uint8)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    size_factor = min(1.0, face_ratio * 12.0)  # want face >= ~8% of frame
    sharp_factor = min(1.0, sharpness / 120.0)  # Laplacian var; >120 is crisp
    score = float(conf) * (0.4 + 0.3 * size_factor + 0.3 * sharp_factor)
    return {
        "score": score,
        "brightness": brightness,
        "sharpness": sharpness,
        "conf": conf,
        "face_ratio": face_ratio,
    }


def _enhance_low_light(frame):
    """CLAHE the L channel when the frame is dim. Returns (frame, enhanced)."""
    gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    if gray_mean >= LOW_LIGHT_MEAN:
        return frame, False
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced, True


def capture_from_camera(
    output_path: str = DEFAULT_OUTPUT,
    camera_index: int = 0,
    auto_capture: bool = False,
    warmup_seconds: float = 1.5,
) -> str | None:
    """Open the webcam, show live face detection + quality overlay, and save
    the BEST recent frame on SPACE (or automatically when quality passes).

    Returns the saved image path on success, None if cancelled or failed.
    """
    from src.face_engine import FaceEngine

    engine = FaceEngine()
    detector = engine.detector

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        console.print("[red]✖ Could not open webcam. Check your camera connection.[/red]")
        return None

    # Request max resolution (webcam defaults are often 640x480, which
    # starves both YuNet detection and Google Lens matching).
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    except Exception:
        pass

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    window_name = "Face Capture - SPACE to capture"
    fullscreen = False
    captured = False
    saved_path = None
    best = None  # (score, frame) of the best buffered frame
    buffer = collections.deque(maxlen=MAX_BUFFER)
    warmup_deadline = cv2.getTickCount() + warmup_seconds * cv2.getTickFrequency()
    auto_fired = False

    console.print(
        "[dim]Camera controls: [SPACE] capture best frame · [A] auto-capture · "
        "[F] fullscreen · [ESC] cancel[/dim]"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = _detect_faces_frame(frame, detector)
        q = _frame_quality(frame, faces)

        # Buffer the RAW frame (enhancement happens at save time).
        buffer.append((q["score"], frame))
        if best is None or q["score"] > best[0]:
            best = (q["score"], frame.copy())

        # Live quality overlay
        display = frame.copy()
        for i, (x, y, w, h, score) in enumerate(faces):
            color = (0, 255, 0) if i == 0 else (0, 200, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                display,
                f"Face {i}: {score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        if faces and q["brightness"] < LOW_LIGHT_MEAN:
            qtext, qcolor = "LOW LIGHT (auto-enhance on capture)", (0, 165, 255)
        elif faces and q["sharpness"] < 60:
            qtext, qcolor = "BLURRY - hold still", (0, 165, 255)
        elif not faces:
            qtext, qcolor = "NO FACE", (0, 0, 255)
        else:
            qtext, qcolor = "QUALITY OK", (0, 255, 0)
        cv2.putText(display, qtext, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, qcolor, 2)
        cv2.putText(
            display,
            f"best score: {best[0]:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        warmed_up = cv2.getTickCount() >= warmup_deadline

        # Auto-capture: buffer warmed, face present, decent score.
        if (
            auto_capture
            and not auto_fired
            and warmed_up
            and faces
            and q["score"] >= 0.45
            and q["brightness"] >= 40
        ):
            key = ord(" ")
            auto_fired = True

        if key == 27:  # ESC
            break
        elif key == ord(" "):  # SPACE -> save the BEST buffered frame
            captured = True
            break
        elif key in (ord("a"), ord("A")):
            auto_capture = not auto_capture
            console.print(f"[dim]auto-capture: {'ON' if auto_capture else 'OFF'}[/dim]")
        elif key == ord("f"):  # Toggle fullscreen
            fullscreen = not fullscreen
            prop = (
                (cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                if fullscreen
                else (cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            )
            cv2.setWindowProperty(window_name, prop[0], prop[1])

    cap.release()
    cv2.destroyAllWindows()

    if captured and best is not None:
        frame_to_save = best[1]
        # Enhance dim frames BEFORE saving so Lens + SFace see a bright face.
        frame_to_save, enhanced = _enhance_low_light(frame_to_save)
        cv2.imwrite(output_path, frame_to_save, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        console.print(
            f"[green]✔ Captured best frame (score {best[0]:.2f}) saved to: "
            f"[bold]{output_path}[/bold][/green]"
            + (" [dim](low-light CLAHE applied)[/dim]" if enhanced else "")
        )
        return output_path

    console.print("[yellow]Capture cancelled.[/yellow]")
    return saved_path
