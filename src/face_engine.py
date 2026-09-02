import hashlib
import os

import cv2
import numpy as np
import requests

# OpenCV Zoo models (Apache-2.0). YuNet = face detection, SFace = 128-d face
# recognition embeddings. Compatible with OpenCV >= 4.5.4 (including OpenCV 5,
# where the legacy CascadeClassifier API was removed).
MODELS_DIR = "models"
YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)
# YuNet is most reliable when the input's largest dimension is moderate.
# Very large photos yield weak confidences, so detection runs on a downscaled
# copy and coordinates are mapped back to the original resolution.
DETECT_MAX_DIM = 1024
SCORE_THRESHOLD = 0.5
# Cosine threshold above which two embeddings are considered the same person
# (OpenCV Zoo reference value for SFace).
SFACE_COSINE_THRESHOLD = 0.363
EMBEDDING_DIM = 128


def _ensure_model(model_path: str, url: str):
    """Fail-safe: auto-download a model file if it is missing."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    print(f"[face_engine] Model not found - downloading to {model_path} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(model_path, "wb") as f:
        f.write(resp.content)
    print("[face_engine] Model downloaded.")


class FaceEngine:
    """Stage 1: face detection, padded cropping, and biometric encoding.

    Encoding uses the SFace model to produce a true 128-dimensional face
    embedding; the biometric hash is SHA-256 over the raw float32 embedding
    bytes, which maps natively onto the EVM `bytes32` type.
    """

    def __init__(self):
        _ensure_model(YUNET_MODEL_PATH, YUNET_MODEL_URL)
        _ensure_model(SFACE_MODEL_PATH, SFACE_MODEL_URL)
        # Input size is set per-image inside process_image()
        self.detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (0, 0),
            score_threshold=SCORE_THRESHOLD,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
        self.last_embedding = None

    def process_image(self, image_path: str, output_crop_path: str = "temp/face_crop.jpg"):
        """Detect the largest face, save a 15%-padded crop, compute the
        embedding hash. Returns (crop_path, face_hash, bbox)."""
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
        face_row = faces[0]

        # Map the detection row (box + landmarks) back to original resolution
        full_res_row = face_row.copy()
        full_res_row[:14] *= 1.0 / scale
        x, y, w, h = (int(v) for v in full_res_row[:4])

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

        # SFace embedding: align the face using the 5 detected landmarks, then
        # extract the 128-d feature vector.
        aligned_face = self.recognizer.alignCrop(image, full_res_row)
        embedding = self.recognizer.feature(aligned_face).flatten().astype(np.float32)
        self.last_embedding = embedding

        # Deterministic biometric hash over the raw embedding bytes (128 x 4B)
        face_hash = "0x" + hashlib.sha256(embedding.tobytes()).hexdigest()

        return output_crop_path, face_hash, (x, y, w, h)

    @staticmethod
    def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Cosine similarity between two SFace embeddings (range -1..1)."""
        a = embedding_a.flatten().astype(np.float64)
        b = embedding_b.flatten().astype(np.float64)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @classmethod
    def same_person(cls, embedding_a: np.ndarray, embedding_b: np.ndarray) -> bool:
        """Cosine-similarity re-identification between two SFace embeddings."""
        return cls.cosine_similarity(embedding_a, embedding_b) >= SFACE_COSINE_THRESHOLD
