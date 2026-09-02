import hashlib
import os

import cv2
import requests

# YuNet (DNN) face detection model from the official OpenCV Zoo (Apache-2.0).
# Compatible with OpenCV >= 4.5.4 (including OpenCV 5, where the legacy
# CascadeClassifier API was removed).
YUNET_MODEL_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
# YuNet is most reliable when the input's largest dimension is moderate.
# Very large photos yield weak confidences, so detection runs on a downscaled
# copy and coordinates are mapped back to the original resolution.
DETECT_MAX_DIM = 1024
SCORE_THRESHOLD = 0.5


class FaceEngine:
    """Stage 1: face detection, padded cropping, and biometric hashing."""

    def __init__(self, model_path: str = YUNET_MODEL_PATH):
        self._ensure_model(model_path)
        # Input size is set per-image inside process_image()
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (0, 0),
            score_threshold=SCORE_THRESHOLD,
        )

    @staticmethod
    def _ensure_model(model_path: str):
        """Fail-safe: auto-download the YuNet model if it is missing."""
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            return
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        print(f"[face_engine] YuNet model not found - downloading to {model_path} ...")
        resp = requests.get(YUNET_MODEL_URL, timeout=60)
        resp.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(resp.content)
        print("[face_engine] YuNet model downloaded.")

    def process_image(self, image_path: str, output_crop_path: str = "temp/face_crop.jpg"):
        """Detect the largest face, save a 15%-padded crop, return (crop_path, face_hash, bbox)."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not decode image at {image_path}")

        # Detect on a downscaled copy for reliable confidences, then rescale
        scale = 1.0
        detect_img = image
        if max(image.shape[:2]) > DETECT_MAX_DIM:
            scale = DETECT_MAX_DIM / max(image.shape[:2])
            detect_img = cv2.resize(image, None, fx=scale, fy=scale)

        self.detector.setInputSize((detect_img.shape[1], detect_img.shape[0]))
        _, faces = self.detector.detect(detect_img)
        if faces is None or len(faces) == 0:
            raise ValueError(
                "No face detected in input image. Ensure clear lighting and a front-facing angle."
            )

        # Each row: x, y, w, h, 5x(landmark x, y), confidence.
        # Select the most prominent face by bounding area.
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        face = faces[0]
        x, y, w, h = (int(v / scale) for v in face[:4])

        # 15% contextual padding around the face for reverse image search accuracy
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)

        cropped = image[y1:y2, x1:x2]
        out_dir = os.path.dirname(output_crop_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_crop_path, cropped)

        # Deterministic biometric hash: normalize the tight face crop to 128x128 grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        normalized_face = cv2.resize(gray[y:y + h, x:x + w], (128, 128))
        face_hash = "0x" + hashlib.sha256(normalized_face.tobytes()).hexdigest()

        return output_crop_path, face_hash, (x, y, w, h)
