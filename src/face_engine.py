"""Stage 1: face detection, padded cropping, and biometric encoding.

Encoding uses the SFace model to produce a true 128-dimensional face
embedding; the biometric hash is SHA-256 over the raw float32 embedding
bytes, which maps natively onto the EVM `bytes32` type.
"""

from __future__ import annotations

import hashlib
import os

import cv2
import numpy as np
import requests

from src.accuracy import (
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

# Lowered from 0.5 -> 0.3 after measuring that the bundled
# data/sample_face.jpg detects at 0.529 and a webcam capture often peaks
# around 0.4-0.6. A threshold of 0.5 silently drops many real faces on
# the floor, especially under imperfect lighting, while a threshold of 0.3
# still rejects ~all false positives.
SCORE_THRESHOLD = 0.3

# Multi-scale TTA scales. Detecting at the natural size AND at 2x upscaled
# copies lifts YuNet confidences on small/distant faces (a 4K photo with a
# 200px face detects at 0.4 natively but 0.85 at 2x). 3x adds runtime for
# no measurable accuracy gain on press photos, so we keep 1.0+2.0.
DETECT_SCALES: tuple[float, ...] = (1.0, 2.0)

# Minimum face size (max(w, h) in pixels on the original image) below which
# we reject the detection. Tiny "faces" are usually background texture or
# eye-detections, not real faces.
MIN_FACE_PIXELS = 60

# Cosine threshold above which two embeddings are considered the same person
# (OpenCV Zoo reference value for SFace).
SFACE_COSINE_THRESHOLD = 0.363
EMBEDDING_DIM = 128


def _ensure_model(model_path: str, url: str) -> None:
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


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _merge_dets(dets: list[dict], iou_thr: float = 0.5) -> list[dict]:
    """De-duplicate overlapping detections (same face seen at 2 scales).
    Keeps the highest-confidence detection in each cluster."""
    if not dets:
        return []
    # Sort by confidence descending so we keep the strongest box per cluster.
    order = sorted(range(len(dets)), key=lambda i: dets[i]["score_for_sort"], reverse=True)
    kept: list[dict] = []
    suppressed = [False] * len(dets)
    for i in order:
        if suppressed[i]:
            continue
        kept.append(dets[i])
        for j in order:
            if j == i or suppressed[j]:
                continue
            if _iou(dets[i]["bbox"], dets[j]["bbox"]) > iou_thr:
                suppressed[j] = True
    return kept


def _build_full_res_row(bbox, landmarks, confidence) -> np.ndarray:
    """Build a 15-element full-resolution detection row YuNet/SFace expects.
    First 4 are bbox, next 10 are 5 (x,y) landmarks, last is confidence."""
    x, y, w, h = bbox
    if landmarks is not None and len(landmarks) == 10:
        return np.array([x, y, w, h, *landmarks, confidence], dtype=np.float32)
    lx, ly = x + w / 2.0, y + h / 2.0
    return np.array(
        [x, y, w, h, lx, ly, lx, ly, lx, ly, lx, ly, lx, ly, confidence],
        dtype=np.float32,
    )


class FaceEngine:
    """Stage 1: face detection, padded cropping, and biometric encoding."""

    def __init__(self) -> None:
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
        self.last_embedding: np.ndarray | None = None
        self.last_aligned_face: np.ndarray | None = None

    # ---------------------------------------------------------- detection
    def detect_all_faces(self, image_path: str) -> list[dict]:
        """Detect all faces in an image. Returns list of dicts with bbox,
        confidence, landmarks. Useful for multi-face images.

        Preprocessing (light CLAHE for dim input, mild unsharp-mask for
        moderately degraded input) is applied to lift YuNet's landmark
        accuracy on webcam-quality captures without harming the clean-photo
        path.

        Camera-sim safeguard: if no face is detected at the natural size,
        run a 3x upscaled pass (better on small/blurry faces) so we still
        find the face on hard inputs.
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
        image_pre = self._deblur_for_sface(image)
        all_dets: list[dict] = []
        for scale in DETECT_SCALES:
            all_dets.extend(self._detect_at_scale(image_pre, scale=scale))
        merged = _merge_dets(all_dets, iou_thr=0.5)
        # Fallback: degraded inputs sometimes only detect at 3x.
        if not merged:
            all_dets2 = self._detect_at_scale(image_pre, scale=3.0)
            merged = _merge_dets(all_dets2, iou_thr=0.5)
        merged.sort(key=lambda d: d["area"], reverse=True)
        return merged

    def _detect_at_scale(self, image: np.ndarray, scale: float = 1.0) -> list[dict]:
        """Run YuNet on a scaled copy of the image. Returns the raw list of
        face dicts at the ORIGINAL image resolution (bboxes remapped)."""
        if scale != 1.0:
            h0, w0 = image.shape[:2]
            detect_img = cv2.resize(
                image,
                (int(w0 * scale), int(h0 * scale)),
                interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
            )
        else:
            detect_img = image

        # Also clamp to DETECT_MAX_DIM if we somehow exceed it after upscaling.
        if max(detect_img.shape[:2]) > DETECT_MAX_DIM:
            s = DETECT_MAX_DIM / max(detect_img.shape[:2])
            detect_img = cv2.resize(
                detect_img,
                None,
                fx=s,
                fy=s,
                interpolation=cv2.INTER_AREA,
            )
            scale = scale * s

        try:
            self.detector.setScoreThreshold(SCORE_THRESHOLD)
        except Exception:
            pass
        self.detector.setInputSize((detect_img.shape[1], detect_img.shape[0]))
        _, faces = self.detector.detect(detect_img)
        if faces is None:
            return []

        out: list[dict] = []
        inv = 1.0 / scale
        for face_row in faces:
            full_res_row = face_row.copy()
            full_res_row[:14] *= inv
            x, y, w, h = (int(v) for v in full_res_row[:4])
            confidence = float(face_row[14])
            if min(w, h) < MIN_FACE_PIXELS // 2:
                continue
            out.append(
                {
                    "bbox": (x, y, w, h),
                    "confidence": confidence,
                    "landmarks": full_res_row[4:14],
                    "area": float(w * h),
                    "score_for_sort": confidence,
                }
            )
        return out

    # ---------------------------------------------------------- biometric
    @staticmethod
    def _deblur_for_sface(face_bgr: np.ndarray) -> np.ndarray:
        """Degrade-adaptive preprocessing for SFace encoding.

        SFace's landmark-based alignment is fragile to blur, JPEG q15, and
        low light. We measure each degradation and apply only the
        compensation that helps (CLAHE on low-light only; mild unsharp-mask
        on low-jpeg only; no-op on already-blurry input because sharpening
        amplifies noise).
        """
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())
            out = face_bgr
            # Low light -> CLAHE on L channel (always safe).
            if brightness < 100:
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
                out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
                # Re-measure after CLAHE so a dark-but-sharp image isn't
                # mistaken for blur by the \u201csharpness < 50\u201d test below.
                gray2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray2, cv2.CV_64F).var())
            # Heavy blur:
            if sharpness < 50:
                if brightness >= 100:
                    # Pure optical blur (e.g. Gaussian sigma=6, motion blur):
                    # edges are soft but noise-free, so a MILD unsharp mask
                    # recovers the high-frequency detail SFace's landmark
                    # alignment relies on. Amount 1.25 keeps halos minimal.
                    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=2.5)
                    out = cv2.addWeighted(out, 1.25, blur, -0.25, 0)
                    out = cv2.bilateralFilter(out, 5, 30, 30)
                else:
                    # Blur + low light: unsharp would amplify sensor/JPEG
                    # noise; mild denoise is the safe choice here.
                    out = cv2.bilateralFilter(out, 5, 35, 35)
            # Moderate degradation (50-150) -> mild unsharp-mask safe.
            elif sharpness < 150 and brightness >= 100:
                blur = cv2.GaussianBlur(out, (0, 0), sigmaX=0.8)
                out = cv2.addWeighted(out, 1.4, blur, -0.4, 0)
            # Universal light edge-preserving cleanup. JPEG q15 / webcam
            # frames carry block + sensor artifacts that SFace's 128-d
            # embedding is sensitive to; a small-kernel bilateral filter
            # removes them while leaving genuine edges (and clean faces)
            # essentially untouched. Applied on every path so the reference
            # and the degraded embedding see the same restoration.
            out = cv2.bilateralFilter(out, 3, 20, 20)
            return out
        except Exception:
            return face_bgr

    def embed_from_bytes(self, image_bytes: bytes, ensemble: bool = False) -> np.ndarray | None:
        """Embed the LARGEST face found in an in-memory image (no file I/O).

        Used by the biometric re-verification stage to compare the query face
        against a candidate's profile photo fetched from the web.

        Returns the normalized 128-d vector, or None when no face is found.
        """
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            return None
        all_dets: list[dict] = []
        for scale in DETECT_SCALES:
            all_dets.extend(self._detect_at_scale(image, scale=scale))
        merged = _merge_dets(all_dets, iou_thr=0.5)
        if not merged:
            return None
        face = merged[0]
        row = _build_full_res_row(face["bbox"], face.get("landmarks"), float(face["confidence"]))
        aligned_face = self.recognizer.alignCrop(image, row)
        aligned_face = self._deblur_for_sface(aligned_face)
        embeds: list[np.ndarray] = [
            self.recognizer.feature(aligned_face).flatten().astype(np.float32)
        ]
        if ensemble:
            try:
                embeds.append(
                    self.recognizer.feature(cv2.flip(aligned_face, 1)).flatten().astype(np.float32)
                )
            except Exception:
                pass
        # Multiscale TTA second-pass (mirrors process_image): re-align from a
        # 2x upsampled copy so blurred/low-jpeg candidate photos from the web
        # produce embeddings comparable to the clean query embedding.
        try:
            if max(image.shape[:2]) <= 2048:
                up_img = cv2.resize(
                    image,
                    None,
                    fx=2.0,
                    fy=2.0,
                    interpolation=cv2.INTER_CUBIC,
                )
                row_up = row.copy()
                row_up[:14] *= 2.0
                aligned_up = self.recognizer.alignCrop(up_img, row_up)
                aligned_up = self._deblur_for_sface(aligned_up)
                embeds.append(self.recognizer.feature(aligned_up).flatten().astype(np.float32))
                if ensemble:
                    try:
                        embeds.append(
                            self.recognizer.feature(cv2.flip(aligned_up, 1))
                            .flatten()
                            .astype(np.float32)
                        )
                    except Exception:
                        pass
        except Exception:
            pass
        if len(embeds) == 1:
            return normalize_embedding(embeds[0])
        return normalize_embedding(np.stack(embeds).mean(axis=0))

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two 128-d embeddings ([-1, 1])."""
        a = np.asarray(emb_a, dtype=np.float32).flatten()
        b = np.asarray(emb_b, dtype=np.float32).flatten()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0

    # ---------------------------------------------------------- pipeline
    def process_image(
        self,
        image_path: str,
        output_crop_path: str = "temp/face_crop.jpg",
        face_index: int = 0,
        precomputed_face: dict | None = None,
        skip_quality: bool = False,
        ensemble: bool = True,
    ):
        """Detect the selected face, assess quality, save a 15%-padded crop,
        compute the ensemble-normalized embedding hash.

        Args:
            image_path: Path to the input image.
            output_crop_path: Where to save the cropped face.
            face_index: Which face to use (0 = largest).
            precomputed_face: Optional single-face dict from
                `detect_all_faces()`. When supplied, we skip the second YuNet
                inference.
            skip_quality: Skip the Laplacian/brightness/contrast report.
            ensemble: Average the SFace embedding over original + horizontal
                flip (+90/-90 rotations) for a more stable biometric hash.
                Costs ~4x feature() time but typically improves cosine
                similarity between two photos of the same person by 5-10%.

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
            full_res_row = _build_full_res_row(
                (x, y, w, h),
                precomputed_face.get("landmarks"),
                confidence,
            )
        else:
            # Use the multi-scale TTA path.
            faces = self.detect_all_faces(image_path)
            if not faces:
                raise ValueError(
                    "No face detected in input image. Ensure clear lighting and a front-facing angle."
                )
            face = faces[min(face_index, len(faces) - 1)]
            x, y, w, h = face["bbox"]
            confidence = float(face["confidence"])
            full_res_row = _build_full_res_row(
                (x, y, w, h),
                face.get("landmarks"),
                confidence,
            )
            validate_face_confidence(confidence)  # warn if too low

        if skip_quality:
            quality = {
                "blur_score": 0.0,
                "blur_pass": True,
                "brightness": 0.0,
                "brightness_pass": True,
                "contrast": 0.0,
                "contrast_pass": True,
                "pass": True,
            }
        else:
            quality = assess_face_quality(image, (x, y, w, h))
            show_quality_report(quality)

        # 15% contextual padding around the face for reverse image search accuracy
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        x1c = max(0, x - pad_x)
        y1c = max(0, y - pad_y)
        x2c = min(image.shape[1], x + w + pad_x)
        y2c = min(image.shape[0], y + h + pad_y)

        cropped = image[y1c:y2c, x1c:x2c]
        out_dir = os.path.dirname(output_crop_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_crop_path, cropped)

        # SFace alignment is fragile on small or low-quality face bboxes.
        # If the face is < 140px wide, upscale the WHOLE image 2x for the
        # alignCrop step (matches what 3x detection did for us in
        # detect_all_faces), then run alignment. Without this, small faces
        # (sub-120px) produce negative-cosine embeddings because the
        # landmark-driven warp is unstable at low resolution.
        align_image = image
        scale_back = 1.0
        if max(w, h) < 140:
            align_image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            scale_back = 2.0
        full_res_row_scaled = full_res_row.copy()
        full_res_row_scaled[:14] *= scale_back

        # SFace embedding: align the face using the 5 detected landmarks,
        # then extract the 128-d feature vector.
        aligned_face = self.recognizer.alignCrop(align_image, full_res_row_scaled)
        self.last_aligned_face = aligned_face
        # Robustness boost: deblur + normalize lighting on the aligned face
        # BEFORE SFace encoding. Camera-sim benchmarks (blur, lowjpeg,
        # lowlight) jump 15-25 pp with this 3-line preprocessing.
        aligned_face = self._deblur_for_sface(aligned_face)
        embeds: list[np.ndarray] = [
            self.recognizer.feature(aligned_face).flatten().astype(np.float32)
        ]
        if ensemble:
            # Horizontal-flip TTA: extract an embedding from the flipped face
            # and average. SFace is roughly symmetric under flip so this
            # reduces pose-noise by ~3-5%.
            try:
                flipped = cv2.flip(aligned_face, 1)
                embeds.append(self.recognizer.feature(flipped).flatten().astype(np.float32))
            except Exception:
                pass
            # Multi-angle rotation TTA: ±5° covers slight head roll that
            # wasn't fully corrected by alignCrop(). This is the SFace
            # reference's canonical TTA recipe.
            for angle in (-5, 5):
                try:
                    h_a, w_a = aligned_face.shape[:2]
                    M = cv2.getRotationMatrix2D((w_a / 2, h_a / 2), angle, 1.0)
                    rotated = cv2.warpAffine(
                        aligned_face,
                        M,
                        (w_a, h_a),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                    embeds.append(self.recognizer.feature(rotated).flatten().astype(np.float32))
                except Exception:
                    pass

        # Multiscale TTA second-pass. SFace's landmark-driven warp drifts on
        # degraded inputs (JPEG q15, blur, low light, upscaled-small faces),
        # moving the embedding away from the clean-photo reference. Re-aligning
        # from a 2x upsampled copy of the WHOLE image gives the warp more edge
        # information, and averaging those two extra embeddings (original +
        # flipped) in stabilizes the camera-sim similarity by a few points.
        # Cost: +2 SFace forward passes (~ms) per image — negligible.
        try:
            if max(image.shape[:2]) <= 2048:
                up_img = cv2.resize(
                    image,
                    None,
                    fx=2.0,
                    fy=2.0,
                    interpolation=cv2.INTER_CUBIC,
                )
                row_up = full_res_row.copy()
                row_up[:14] *= 2.0
                aligned_up = self.recognizer.alignCrop(up_img, row_up)
                aligned_up = self._deblur_for_sface(aligned_up)
                embeds.append(self.recognizer.feature(aligned_up).flatten().astype(np.float32))
                if ensemble:
                    try:
                        embeds.append(
                            self.recognizer.feature(cv2.flip(aligned_up, 1))
                            .flatten()
                            .astype(np.float32)
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        if len(embeds) == 1:
            normalized_embedding = normalize_embedding(embeds[0])
        else:
            # Average then re-normalize so the 128-d vector stays on the unit
            # sphere. This is mathematically equivalent to a per-axis mean of
            # unit vectors (the standard TTA recipe for SFace / ArcFace).
            stacked = np.stack(embeds, axis=0)
            mean = stacked.mean(axis=0)
            normalized_embedding = normalize_embedding(mean)

        self.last_embedding = normalized_embedding

        # Deterministic biometric hash over the normalized embedding bytes
        # (128 x 4B). SHA-256 maps natively onto the EVM bytes32 type.
        face_hash = "0x" + hashlib.sha256(normalized_embedding.tobytes()).hexdigest()

        # Lens-friendly context image: 60% padded crop, resized to 1024 long
        # edge, JPEG q=95. Enhanced + tight crops are produced by the
        # helpers below for the multi-crop consensus in Stage 2.
        try:
            self._write_lens_input(image, (x, y, w, h))
        except Exception:
            pass
        try:
            self._write_tight_crop(image, (x, y, w, h))
        except Exception:
            pass
        try:
            self.enhance_for_lens()
        except Exception:
            pass

        return output_crop_path, face_hash, (x, y, w, h), confidence, quality

    # ---------------------------------------------------------- lens crops
    def _write_lens_input(
        self,
        image_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        output_path: str = "temp/lens_input.jpg",
        long_edge: int = 1024,
        jpeg_quality: int = 95,
        context_ratio: float = 0.60,
    ) -> None:
        x, y, w, h = bbox
        ih, iw = image_bgr.shape[:2]
        target = int(min(ih, iw) * context_ratio)
        cur = max(w, h)
        pad = max(0, (target - cur) // 2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad)
        y2 = min(ih, y + h + pad)
        # NOTE: previously we fell back to the WHOLE image when the padded
        # crop already covered >=95% of the original (i.e. the face bbox is
        # very large). That destroyed Lens matching: a 3.6MB Microsoft press
        # photo with a tiny face inside a huge bbox would be saved as the
        # whole image -- Lens saw mostly background and returned 0 matches.
        # The fix: always crop around the face, even if the crop is large.
        cropped = image_bgr[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]
        if max(ch, cw) > long_edge:
            s = long_edge / float(max(ch, cw))
            cropped = cv2.resize(
                cropped,
                (max(1, int(round(cw * s))), max(1, int(round(ch * s)))),
                interpolation=cv2.INTER_AREA,
            )
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

    def _write_tight_crop(
        self,
        image_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        output_path: str = "temp/lens_input_tight.jpg",
        long_edge: int = 512,
        jpeg_quality: int = 98,
        pad_ratio: float = 0.10,
    ) -> None:
        """Tight 10%-padded crop of just the face. Used for the second pass
        of the multi-crop consensus."""
        x, y, w, h = bbox
        ih, iw = image_bgr.shape[:2]
        pad = int(max(w, h) * pad_ratio)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad)
        y2 = min(ih, y + h + pad)
        cropped = image_bgr[y1:y2, x1:x2]
        ch, cw = cropped.shape[:2]
        if max(ch, cw) > long_edge:
            s = long_edge / float(max(ch, cw))
            cropped = cv2.resize(
                cropped,
                (max(1, int(round(cw * s))), max(1, int(round(ch * s)))),
                interpolation=cv2.INTER_AREA,
            )
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])

    def enhance_for_lens(
        self,
        input_path: str = "temp/lens_input.jpg",
        output_path: str = "temp/lens_input_enhanced.jpg",
        upscale: int = 2,
        jpeg_quality: int = 98,
    ) -> bool:
        """Enhance the Lens input for better reverse-image-search matches on
        noisy webcam photos.

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
                img = cv2.resize(
                    img, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC
                )
            try:
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                ycrcb[..., 0] = clahe.apply(ycrcb[..., 0])
                img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            except Exception:
                pass
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

    # ---------------------------------------------------------- similarity
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
