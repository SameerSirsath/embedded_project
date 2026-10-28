import queue
import re
import threading


class ReadingTTS:
    """Threaded TTS with pause / resume / stop + alert ducking."""

    def __init__(self, rate=170, voice=None):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        if voice:
            for v in self.engine.getProperty("voices"):
                if voice.lower() in v.name.lower():
                    self.engine.setProperty("voice", v.id)
                    break
        self.q        = queue.Queue()
        self.paused   = threading.Event()
        self.stopped  = threading.Event()
        self._current = None
        self._worker  = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # --------- public API ---------------------------------------
    def speak(self, text, interrupt=False):
        if interrupt:
            self.stop()
            self.stopped.clear()
        for sent in self.split(text):
            self.q.put(sent)

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()

    def stop(self):
        self.stopped.set()
        with self.q.mutex:
            self.q.queue.clear()
        self._current = None

    def alert(self, message):
        """Duck reading, speak alert, resume."""
        was_paused = self.paused.is_set()
        self.paused.set()
        try:
            self.engine.say(message)
            self.engine.runAndWait()
        finally:
            if not was_paused:
                self.paused.clear()

    @staticmethod
    def split(text):
        text = re.sub(r"\s+", " ", text).strip()
        return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    # --------- worker -------------------------------------------
    def _run(self):
        while not self.stopped.is_set():
            try:
                sent = self.q.get(timeout=0.2)
            except queue.Empty:
                self._current = None
                continue
            while self.paused.is_set() and not self.stopped.is_set():
                self.paused.wait(0.1)
            if self.stopped.is_set():
                break
            self._current = sent
            try:
                self.engine.say(sent)
                self.engine.runAndWait()
            except Exception as e:
                print(f"[TTS] error: {e}")
            self._current = None
