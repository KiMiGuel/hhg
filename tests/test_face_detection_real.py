# -*- coding: utf-8 -*-
"""Real face-detection tests against the downloaded internet image corpus.

Uses the images already downloaded to ``data/internet_test/`` (no network at
test time). Face images must yield >=1 detection and a valid 66-char biometric
hash; non-face controls must yield zero detections.

Skipped automatically when the corpus or the YuNet/SFace models are missing.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.face_engine import FaceEngine  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS = os.path.join(ROOT, "data", "internet_test")

FACE_IMAGES = [f"portrait_{i}.jpg" for i in range(1, 9)]
NONFACE_IMAGES = ["landscape.jpg", "random_1.jpg", "random_2.jpg"]

pytestmark = pytest.mark.skipif(
    not os.path.isdir(CORPUS) or not os.listdir(CORPUS),
    reason="internet image corpus (data/internet_test) not downloaded",
)


@pytest.fixture(scope="module")
def engine():
    return FaceEngine()


def _valid_hash(face_hash) -> bool:
    if not isinstance(face_hash, str) or not face_hash.startswith("0x"):
        return False
    body = face_hash[2:]
    return len(body) == 64 and all(c in "0123456789abcdef" for c in body.lower())


class TestFaceDetectionOnInternetImages:
    @pytest.mark.parametrize("name", FACE_IMAGES)
    def test_face_image_detected_and_hashed(self, engine, name):
        path = os.path.join(CORPUS, name)
        if not os.path.exists(path):
            pytest.skip(f"corpus image missing: {name}")
        faces = engine.detect_all_faces(path)
        assert len(faces) >= 1, f"{name}: expected >=1 face, got 0"
        assert faces[0]["confidence"] > 0.5, f"{name}: low confidence {faces[0]['confidence']}"
        crop_path = os.path.join(ROOT, "temp", f"_pytest_{name}")
        _, face_hash, bbox, confidence, quality = engine.process_image(
            path, output_crop_path=crop_path, face_index=0)
        assert _valid_hash(face_hash), f"{name}: invalid biometric hash {face_hash!r}"
        assert len(bbox) == 4 and all(v >= 0 for v in bbox), f"{name}: bad bbox {bbox}"

    @pytest.mark.parametrize("name", NONFACE_IMAGES)
    def test_nonface_image_returns_no_faces(self, engine, name):
        path = os.path.join(CORPUS, name)
        if not os.path.exists(path):
            pytest.skip(f"corpus image missing: {name}")
        faces = engine.detect_all_faces(path)
        assert len(faces) == 0, (
            f"{name}: false-positive detection(s) on non-face image: "
            f"{[(f['bbox'], f['confidence']) for f in faces]}")

    def test_detection_is_deterministic(self, engine):
        path = os.path.join(CORPUS, FACE_IMAGES[0])
        if not os.path.exists(path):
            pytest.skip(f"corpus image missing: {FACE_IMAGES[0]}")
        faces_a = engine.detect_all_faces(path)
        faces_b = engine.detect_all_faces(path)
        assert len(faces_a) == len(faces_b)
        if faces_a:
            assert faces_a[0]["bbox"] == faces_b[0]["bbox"]

    def test_hash_changes_for_different_people(self, engine):
        paths = [os.path.join(CORPUS, n) for n in FACE_IMAGES[:3]
                 if os.path.exists(os.path.join(CORPUS, n))]
        if len(paths) < 2:
            pytest.skip("need >=2 corpus face images")
        hashes = []
        for i, p in enumerate(paths):
            _, face_hash, *_ = engine.process_image(
                p, output_crop_path=os.path.join(ROOT, "temp", f"_pytest_diff_{i}.jpg"),
                face_index=0)
            hashes.append(face_hash)
        assert len(set(hashes)) == len(hashes), "distinct faces produced identical hashes"
