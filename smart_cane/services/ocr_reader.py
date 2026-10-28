import time

import cv2
import numpy as np


# --------- pure helpers (unit-testable) --------------------------
def sharpest_of(frames):
    """Return (frame, focus_score) with max Laplacian variance."""
    best, best_score = frames[0], -1.0
    for f in frames:
        gray  = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score > best_score:
            best, best_score = f, score
    return best, best_score


def preprocess(img, mode="book"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "screen":
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        gray = cv2.createCLAHE(clipLimit=2.0,
                               tileGridSize=(8, 8)).apply(gray)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )
    else:
        gray = cv2.createCLAHE(clipLimit=3.0,
                               tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 15,
        )
        thresh = cv2.resize(thresh, None, fx=1.5, fy=1.5,
                            interpolation=cv2.INTER_CUBIC)
    return thresh


def _largest_gap(xs_sorted):
    if len(xs_sorted) < 2:
        return None
    gaps = [(xs_sorted[i], xs_sorted[i + 1])
            for i in range(len(xs_sorted) - 1)]
    return max(gaps, key=lambda g: g[1] - g[0])


def _join_lines(lines):
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out.endswith("-"):
            out = out[:-1] + line
        else:
            out += " " + line
    return out.strip()


def order_blocks(blocks, img_shape):
    """Sort OCR blocks into human reading order; handles 1/2 columns."""
    H, W = img_shape[:2]
    blocks = [b for b in blocks
              if b["conf"] > 0.6 and len(b["text"].strip()) > 1]
    blocks = [b for b in blocks
              if 0.06 * H < b["bbox"][1] < 0.94 * H]
    if not blocks:
        return ""
    xs  = sorted(b["bbox"][0] for b in blocks)
    gap = _largest_gap(xs)
    if gap and (gap[1] - gap[0]) > 0.15 * W:
        mid   = (gap[0] + gap[1]) / 2.0
        left  = sorted([b for b in blocks if b["bbox"][0] <  mid],
                       key=lambda b: b["bbox"][1])
        right = sorted([b for b in blocks if b["bbox"][0] >= mid],
                       key=lambda b: b["bbox"][1])
        ordered = left + right
    else:
        ordered = sorted(blocks,
                         key=lambda b: (b["bbox"][1] // 20, b["bbox"][0]))
    return _join_lines([b["text"] for b in ordered])


# --------- main reader class -------------------------------------
class PageReader:
    def __init__(self, capture_fn, engine="rapidocr", mode="book",
                 burst_frames=3, burst_delay=0.15):
        self.capture_fn   = capture_fn
        self.mode         = mode
        self.burst_frames = burst_frames
        self.burst_delay  = burst_delay
        self.bookmarks    = None
        self._ocr         = self._init_engine(engine)

    @staticmethod
    def _init_engine(engine):
        if engine == "rapidocr":
            try:
                from rapidocr_onnxruntime import RapidOCR
                return RapidOCR()
            except ImportError:
                print("[OCR] rapidocr missing; falling back to tesseract")
        import pytesseract
        return pytesseract

    def grab(self):
        frames = []
        for _ in range(self.burst_frames):
            frames.append(self.capture_fn())
            time.sleep(self.burst_delay)
        best, _ = sharpest_of(frames)
        return best

    def _rapidocr_blocks(self, img):
        result, _ = self._ocr(img)
        if not result:
            return []
        blocks = []
        for box, text, conf in result:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            blocks.append({
                "text": text,
                "bbox": [min(xs), min(ys),
                         max(xs) - min(xs), max(ys) - min(ys)],
                "conf": float(conf),
            })
        return blocks

    def extract_text(self, img):
        proc = preprocess(img, mode=self.mode)
        if hasattr(self._ocr, "process"):
            return order_blocks(self._rapidocr_blocks(proc), proc.shape)
        return self._ocr.image_to_string(proc)

    @staticmethod
    def page_hash(img):
        from .bookmarks import BookmarkStore
        return BookmarkStore.page_hash(img)
