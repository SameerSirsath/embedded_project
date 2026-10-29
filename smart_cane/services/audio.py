import os
import queue
import re
import threading


class ReadingTTS:
    """
    Threaded TTS with pause / resume / stop + alert ducking.
    Supports Windows SAPI (via COM with CoInitialize) and Linux (espeak-ng / pyttsx3).
    """

    def __init__(self, rate=170, voice=None):
        self.rate = rate
        self.voice = voice
        self.q = queue.Queue()
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self._current = None
        self._worker = threading.Thread(target=self._run, daemon=True)
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
            # Immediate synchronous alert in current thread
            if os.name == "nt":
                try:
                    import pythoncom
                    import win32com.client
                    pythoncom.CoInitialize()
                    sapi = win32com.client.Dispatch("SAPI.SpVoice")
                    sapi.Rate = 1
                    sapi.Volume = 100
                    sapi.Speak(message)
                    return
                except Exception:
                    pass
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate", self.rate)
            eng.say(message)
            eng.runAndWait()
        except Exception:
            pass
        finally:
            if not was_paused:
                self.paused.clear()

    @staticmethod
    def split(text):
        text = re.sub(r"\s+", " ", text).strip()
        return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    # --------- worker -------------------------------------------
    def _run(self):
        engine = None
        sapi_voice = None

        if os.name == "nt":
            try:
                import pythoncom
                import win32com.client
                pythoncom.CoInitialize()
                sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                sapi_voice.Rate = 1
                sapi_voice.Volume = 100
            except Exception:
                sapi_voice = None

        if sapi_voice is None:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                if self.voice:
                    for v in engine.getProperty("voices"):
                        if self.voice.lower() in v.name.lower():
                            engine.setProperty("voice", v.id)
                            break
            except Exception:
                engine = None

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
                if sapi_voice is not None:
                    sapi_voice.Speak(sent)
                elif engine is not None:
                    engine.say(sent)
                    engine.runAndWait()
                else:
                    import subprocess
                    subprocess.run(["espeak-ng", sent], timeout=4.0)
            except Exception as e:
                print(f"[TTS] error speaking '{sent}': {e}")
            finally:
                self._current = None
                self.q.task_done()
