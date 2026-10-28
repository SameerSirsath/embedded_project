import hashlib
import json
import os
import time

import cv2


class BookmarkStore:
    def __init__(self, path="~/.smart_cane_bookmarks.json"):
        self.path = os.path.expanduser(path)
        self.data = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[bookmarks] save failed: {e}")

    @staticmethod
    def page_hash(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
               if img.ndim == 3 else img
        small = cv2.resize(gray, (16, 16),
                           interpolation=cv2.INTER_AREA)
        return hashlib.md5(small.tobytes()).hexdigest()

    def save(self, page_hash, sentence_index, book_id="default"):
        self.data[book_id] = {
            "page_hash":      page_hash,
            "sentence_index": int(sentence_index),
            "timestamp":      time.time(),
        }
        self._save()

    def resume(self, page_hash, book_id="default"):
        entry = self.data.get(book_id)
        if entry and entry.get("page_hash") == page_hash:
            return int(entry.get("sentence_index", 0))
        return 0
