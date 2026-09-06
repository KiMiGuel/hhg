# -*- coding: utf-8 -*-
"""Exact-image matching across platforms via perceptual hashing.

Google Lens returns a thumbnail for every visual match. By perceptually
hashing (dHash, 64-bit) the uploaded image and each match thumbnail, we can
detect when the EXACT same image (or a trivially recompressed variant)
appears on Instagram, YouTube, Facebook, X, LinkedIn, news sites, etc.

dHash is robust to JPEG recompression, minor resizing, and watermarking;
Hamming distance <= 10 of 64 bits is a near-certain duplicate.
"""
from __future__ import annotations

import cv2
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.biometric_verify import fetch_image  # UA + content-type guard

DUPLICATE_THRESHOLD = 12   # of 64 bits (raised from 10: platform images are heavily cropped + recompressed)
MAX_FETCH = 20             # thumbnails to check per query (raised from 12 for more cross-platform coverage)
MAX_WORKERS = 8            # parallel thumbnail fetches (was 6)


def _dhash_bits(gray: np.ndarray, hash_size: int = 8) -> int:
    """dHash: horizontal gradient bits. Returns a 64-bit int."""
    resized = cv2.resize(gray, (hash_size + 1, hash_size),
                         interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def dhash_from_image(image_bgr: np.ndarray) -> int | None:
    if image_bgr is None or not image_bgr.size:
        return None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return _dhash_bits(gray)


def dhash_from_bytes(data: bytes) -> int | None:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return dhash_from_image(img)


def dhash_from_path(path: str) -> int | None:
    img = cv2.imread(path)
    return dhash_from_image(img)


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _fetch_thumbnail_dhash(url: str) -> int | None:
    data = fetch_image(url, timeout=10)
    if not data:
        return None
    return dhash_from_bytes(data)


def find_exact_matches(
    query_image_path: str,
    matches,
    threshold: int = DUPLICATE_THRESHOLD,
    max_fetch: int = MAX_FETCH,
    max_workers: int = MAX_WORKERS,
) -> list[dict]:
    """Find visual matches whose thumbnail is the SAME image as the query.

    Args:
        query_image_path: the uploaded/captured image on disk.
        matches: LensMatch-like objects with ``thumbnail``, ``link``,
            ``title``, ``platform`` fields (dicts also accepted).
        threshold: max Hamming distance (of 64) to call it the same image.

    Returns matches sorted by Hamming distance, e.g.
        [{"platform": "youtube", "link": "...", "hamming": 3, ...}]
    """
    query_hash = dhash_from_path(query_image_path)
    if query_hash is None:
        return []

    candidates = []
    seen_thumbs = set()
    for m in matches or []:
        thumb = (m.get("thumbnail") if isinstance(m, dict)
                 else getattr(m, "thumbnail", "")) or ""
        if not thumb or thumb in seen_thumbs:
            continue
        seen_thumbs.add(thumb)
        candidates.append({
            "title": (m.get("title") if isinstance(m, dict)
                      else getattr(m, "title", "")) or "",
            "link": (m.get("link") if isinstance(m, dict)
                     else getattr(m, "link", "")) or "",
            "platform": (m.get("platform") if isinstance(m, dict)
                         else getattr(m, "platform", "")) or "",
            "thumbnail": thumb,
        })
        if len(candidates) >= max_fetch:
            break

    def _check(cand: dict) -> dict | None:
        h = _fetch_thumbnail_dhash(cand["thumbnail"])
        if h is None:
            return None
        dist = hamming_distance(query_hash, h)
        if dist <= threshold:
            return {**cand, "hamming": dist}
        return None

    exact: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_check, c): c for c in candidates}
        for fut in as_completed(futures):
            try:
                hit = fut.result()
            except Exception:
                hit = None
            if hit:
                exact.append(hit)
    exact.sort(key=lambda d: d["hamming"])
    return exact
