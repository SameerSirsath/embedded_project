import threading

from ..config import Config


class ModeManager:
    def __init__(self, page_reader, tts, button,
                 detector_controller=None):
        self.reader   = page_reader
        self.tts      = tts
        self.button   = button
        self.detector = detector_controller
        self.mode     = "navigation"
        self._page_hash = None

    def start(self):
        self.button.on_single = self._on_single
        self.button.on_double = self._on_double
        self.button.on_triple = self._on_triple
        self.button.on_long   = self._on_long

    # --------- handlers -----------------------------------------
    def _on_single(self):
        if self.mode == "navigation":
            self._enter_reading()
        else:
            if self.tts.paused.is_set():
                self.tts.resume()
                self.tts.speak("Resuming.")
            else:
                self.tts.pause()
                self.tts.speak("Paused.")

    def _on_double(self):
        if self.mode == "reading":
            self.tts.stop()
            self.tts.speak("Stopped.")

    def _on_triple(self):
        if self.mode == "reading":
            self._read_page()

    def _on_long(self):
        if self.mode == "reading":
            self._exit_reading()

    # --------- transitions --------------------------------------
    def _enter_reading(self):
        self.mode = "reading"
        if self.detector:
            self.detector.set_fps(Config.READING_FPS)
        self.tts.speak(
            "Reading mode. Press once to read, twice to stop, "
            "triple to read next page, hold to exit."
        )
        threading.Timer(3.0, self._read_page).start()

    def _exit_reading(self):
        self.tts.stop()
        self.tts.speak("Navigation mode.")
        self.mode = "navigation"
        if self.detector:
            self.detector.set_fps(Config.NAV_FPS)

    # --------- reading ------------------------------------------
    def _read_page(self):
        if self.mode != "reading":
            return
        try:
            img = self.reader.grab()
        except Exception as e:
            print(f"[ModeManager] capture error: {e}")
            self.tts.speak("I couldn't capture the page. Try again.")
            return
        try:
            text = self.reader.extract_text(img)
        except Exception as e:
            print(f"[ModeManager] ocr error: {e}")
            self.tts.speak("OCR failed. More light, or hold still.")
            return
        if not text.strip():
            self.tts.speak("I couldn't find any text on that page.")
            return

        self._page_hash = self.reader.page_hash(img)
        sentences       = self.tts.split(text)
        start = 0
        if self.reader.bookmarks:
            start = self.reader.bookmarks.resume(self._page_hash)
        if 0 < start < len(sentences):
            self.tts.speak(f"Resuming from sentence {start + 1}.")
        for i in range(start, len(sentences)):
            self.tts.q.put(sentences[i])
            if self.reader.bookmarks:
                self.reader.bookmarks.save(self._page_hash, i + 1)
