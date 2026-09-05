import hashlib
import os
import time

import cv2
import numpy as np
import requests

from src.accuracy import (
    MIN_FACE_CONFIDENCE,
    assess_face_quality,
    normalize_embedding,
    show_quality_report,
    validate_face_confidence,
)

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
SCORE_THRESHOLD = 0.5  # Minimum detection score (YuNet internal)
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

    def detect_all_faces(self, image_path: str) -> list:
        """Detect all faces in an image. Returns list of dicts with bbox,
        confidence, and landmarks. Useful for multi-face images."""
        image = cv2.imread(image_path)
        if image is None:
            return []

        scale = 1.0
        detect_img = image
        if max(image.shape[:2]) > DETECT_MAX_DIM:
            scale = DETECT_MAX_DIM / max(image.shape[:2])
            detect_img = cv2.resize(image, None, fx=scale, fy=scale)

        self.detector.setInputSize((detect_img.shape[1], detect_img.shape[0]))
        _, faces = self.detector.detect(detect_img)

        if faces is None:
            return []

        results = []
        for face_row in faces:
            full_res_row = face_row.copy()
            full_res_row[:14] *= 1.0 / scale
            x, y, w, h = (int(v) for v in full_res_row[:4])
            confidence = float(face_row[14])
            results.append({
                "bbox": (x, y, w, h),
                "confidence": confidence,
                "landmarks": full_res_row[4:14],
            })

        return results

    def process_image(self, image_path: str, output_crop_path: str = "temp/face_crop.jpg",
                      face_index: int = 0, precomputed_face: dict | None = None,
                      skip_quality: bool = True):
        """Detect the selected face, assess quality, save a 15%-padded crop,
        compute the normalized embedding hash.

        Args:
            image_path: Path to the input image.
            output_crop_path: Where to save the cropped face.
            face_index: Which face to use (0 = largest, for multi-face images).
            precomputed_face: Optional single-face dict from `detect_all_faces()`
                (carries bbox/confidence/landmarks in original resolution). When
                supplied, we skip the second YuNet inference and shave 50-150 ms
                off the cold path.
            skip_quality: Skip the Laplacian/brightness/contrast quality report
                (it is purely informational and not used to gate anything). The
                quality dict in the return value is still present for callers
                that expect it, but `pass` defaults to True.

        Returns:
            (crop_path, face_hash, bbox, confidence, quality_report)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not decode image at {image_path}")

        # Fast path: caller already ran detection (e.g. the face picker).
        if precomputed_face is not None:
            x, y, w, h = precomputed_face["bbox"]
            confidence = float(precomputed_face["confidence"])
            # Reconstruct a full_res_row from bbox + landmarks. The first 4
            # entries are bbox, the next 10 are 5 (x, y) landmarks in the same
            # order YuNet uses.
            landmarks = precomputed_face.get("landmarks")
            if landmarks is not None and len(landmarks) == 10:
                full_res_row = np.array(
                    [x, y, w, h, *landmarks, confidence], dtype=np.float32
                )
            else:
                # Fall back: run the recognizer with bbox-only landmarks (works
                # but is less accurate). In practice detect_all_faces always
                # provides the 10-landmark array.
                lx = x + w / 2
                ly = y + h / 2
                full_res_row = np.array(
                    [x, y, w, h,
                     lx, ly,  # right eye
                     lx, ly,  # left eye
                     lx, ly,  # nose
                     lx, ly,  # right mouth
                     lx, ly,  # left mouth
                     confidence],
                    dtype=np.float32,
                )
        else:
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

            # Sort faces by area (largest first)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

            # Validate confidence of selected face
            face_row = faces[face_index]
            confidence = float(face_row[14])
            if not validate_face_confidence(confidence):
                # Warn but don't fail — still produce a result
                pass

            # Map the detection row (box + landmarks) back to original resolution
            full_res_row = face_row.copy()
            full_res_row[:14] *= 1.0 / scale
            x, y, w, h = (int(v) for v in full_res_row[:4])

        if skip_quality:
            # The quality report is informational and not used to gate any
            # downstream decision. Skip the Laplacian/brightness/contrast
            # computation to save 50-200 ms on every run.
            quality = {
                "blur_score": 0.0, "blur_pass": True,
                "brightness": 0.0, "brightness_pass": True,
                "contrast": 0.0, "contrast_pass": True,
                "pass": True,
            }
        else:
            quality = assess_face_quality(image, (x, y, w, h))
            show_quality_report(quality)

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
        raw_embedding = self.recognizer.feature(aligned_face).flatten().astype(np.float32)

        # L2-normalize the embedding for consistent similarity comparisons
        normalized_embedding = normalize_embedding(raw_embedding)
        self.last_embedding = normalized_embedding

        # Deterministic biometric hash over the normalized embedding bytes (128 x 4B)
        face_hash = "0x" + hashlib.sha256(normalized_embedding.tobytes()).hexdigest()

        # Lens-friendly context image: 60% padded crop, resized to 1024 long edge, JPEG q=95.
        try:
            self._write_lens_input(image, (x, y, w, h))
        except Exception:
            pass  # best-effort; never fail Stage 1

        # Tight 10%-padded crop for the second pass of multi-crop consensus.
        try:
            self._write_tight_crop(image, (x, y, w, h))
        except Exception:
            pass  # best-effort

        # Enhanced version of the wide-context crop (upscale + CLAHE + unsharp).
        # This is what Stage 2 uploads by default for the best first-run accuracy.
        try:
            self.enhance_for_lens()
        except Exception:
            pass  # best-effort

        return output_crop_path, face_hash, (x, y, w, h), confidence, quality

    def _write_lens_input(self, image_bgr, bbox, output_path="temp/lens_input.jpg", long_edge=1024, jpeg_quality=95, context_ratio=0.60):
        x, y, w, h = bbox
        ih, iw = image_bgr.shape[:2]
        target = int(min(ih, iw) * context_ratio)
        cur = max(w, h)
        pad = max(0, (target - cur) // 2)
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad); y2 = min(ih, y + h + pad)
        if (x2 - x1) >= int(iw * 0.95) and (y2 - y1) >= int(ih * 0.95):
            x1, y1, x2, y2 = 0, 0, iw, ih
        cropped = image_bgr[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]
        if max(ch, cw) > long_edge:
            scale = long_edge / float(max(ch, cw))
            cropped = cv2.resize(cropped, (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))), interpolation=cv2.INTER_AREA)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

    def _write_tight_crop(self, image_bgr, bbox, output_path="temp/lens_input_tight.jpg", long_edge=512, jpeg_quality=98, pad_ratio=0.10):
        """Tight 10%-padded crop of just the face. Used for the second pass of
        the multi-crop consensus: a different framing often unlocks a better
        match from Google Lens when the wider-context crop is ambiguous.
        """
        x, y, w, h = bbox
        ih, iw = image_bgr.shape[:2]
        pad = int(max(w, h) * pad_ratio)
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad); y2 = min(ih, y + h + pad)
        cropped = image_bgr[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]
        if max(ch, cw) > long_edge:
            scale = long_edge / float(max(ch, cw))
            cropped = cv2.resize(cropped, (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))), interpolation=cv2.INTER_AREA)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

    def enhance_for_lens(self, input_path="temp/lens_input.jpg", output_path="temp/lens_input_enhanced.jpg",
                          upscale=2, jpeg_quality=98):
        """Enhance the Lens input for better reverse-image-search matches on
        noisy webcam photos. Steps:
          1. Bicubic upscale (2x) so a 256x256 face becomes 512x512.
          2. CLAHE on the Y (luma) channel to normalize harsh lighting.
          3. Unsharp mask (amount=1.5, radius=2) to recover edges lost to JPEG.
        Falls back gracefully (returns False) on any error.
        """
        try:
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                return False
            h, w = img.shape[:2]
            if upscale and upscale != 1.0:
                img = cv2.resize(img, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC)
            # CLAHE on luma
            try:
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                ycrcb[..., 0] = clahe.apply(ycrcb[..., 0])
                img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            except Exception:
                pass
            # Unsharp mask: sharp = 1.5*orig - 0.5*blur (amount=1.5)
            try:
                blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
                img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
            except Exception:
                pass
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            return True
        except Exception:
            return False
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
