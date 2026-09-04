"""Camera capture module: real-time webcam face detection with capture.

Press SPACE to capture the current frame (with the largest face).
Press ESC to cancel.
Press 'f' to toggle fullscreen preview.
"""

import os
import cv2

from rich.console import Console

console = Console()

CAPTURE_DIR = "data"
DEFAULT_OUTPUT = os.path.join(CAPTURE_DIR, "captured_face.jpg")


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


def capture_from_camera(output_path: str = DEFAULT_OUTPUT, camera_index: int = 0) -> str | None:
    """Open the webcam, show live face detection, and capture on SPACE.

    Returns the saved image path on success, None if cancelled or failed.
    """
    from src.face_engine import FaceEngine

    engine = FaceEngine()
    # Reuse the YuNet detector from FaceEngine
    detector = engine.detector

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        console.print("[red]✖ Could not open webcam. Check your camera connection.[/red]")
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fullscreen = False
    captured = False
    frame_to_save = None

    console.print("[dim]Camera controls: [SPACE] capture · [F] fullscreen · [ESC] cancel[/dim]")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect faces in real-time
        faces = _detect_faces_frame(frame, detector)

        # Draw bounding boxes and confidence
        display = frame.copy()
        for i, (x, y, w, h, score) in enumerate(faces):
            color = (0, 255, 0) if i == 0 else (0, 200, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            label = f"Face {i}: {score:.2f}"
            cv2.putText(display, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Status overlay
        face_count = len(faces)
        status = f"Faces: {face_count} | SPACE capture | F fullscreen | ESC cancel"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show preview
        window_name = "Face Capture - Press SPACE to capture"
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(" "):  # SPACE
            frame_to_save = frame.copy()
            captured = True
            break
        elif key == ord("f"):  # Toggle fullscreen
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

    if captured and frame_to_save is not None:
        # Save the full frame. Stage 1 owns final face selection, quality checks,
        # SFace alignment, and crop generation. Saving the full frame preserves
        # hair/shoulders/background context for Google Lens accuracy.
        cv2.imwrite(output_path, frame_to_save)
        console.print(f"[green]✔ Captured and saved to: [bold]{output_path}[/bold][/green]")
        return output_path

    console.print("[yellow]Capture cancelled.[/yellow]")
    return None
