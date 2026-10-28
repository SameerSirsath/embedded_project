import cv2
import numpy as np

from smart_cane.services.ocr_reader import (
    sharpest_of, preprocess, order_blocks, _join_lines,
)
from smart_cane.services.bookmarks import BookmarkStore


def test_sharpest_of_picks_blurry_vs_sharp():
    sharp  = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    blurry = cv2.GaussianBlur(sharp, (31, 31), 0)
    best, _ = sharpest_of([blurry, sharp])
    assert best is sharp


def test_preprocess_returns_binary_image():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    out = preprocess(img, mode="book")
    assert out.ndim == 2
    assert set(np.unique(out)).issubset({0, 255})


def test_join_lines_dehyphenates():
    assert _join_lines(["obsta-", "cle ahead"]) == "obstacle ahead"


def test_order_blocks_single_column():
    H, W = 500, 400
    blocks = [
        {"text": "Second line", "bbox": [50, 200, 200, 30], "conf": 0.9},
        {"text": "First line",  "bbox": [50, 100, 200, 30], "conf": 0.9},
    ]
    text = order_blocks(blocks, (H, W, 3))
    assert text.startswith("First line")


def test_page_hash_stable_for_identical_frames():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    b = a.copy()
    assert BookmarkStore.page_hash(a) == BookmarkStore.page_hash(b)
