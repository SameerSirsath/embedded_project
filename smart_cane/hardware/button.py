import threading
import time


class ButtonController:
    """Single / double / triple / long-press dispatcher."""

    def __init__(self, pin=17, on_single=None, on_double=None,
                 on_triple=None, on_long=None,
                 long_press_sec=1.5, double_gap=0.4):
        self.on_single = on_single
        self.on_double = on_double
        self.on_triple = on_triple
        self.on_long   = on_long
        self.long_press_sec = long_press_sec
        self.double_gap     = double_gap
        self._count       = 0
        self._press_start = 0.0
        self._timer       = None
        self._lock        = threading.Lock()

        try:
            from gpiozero import Button
            self.btn = Button(pin, pull_up=True, bounce_time=0.05)
            self.btn.when_pressed  = self._on_press
            self.btn.when_released = self._on_release
        except Exception as e:
            self.btn = None
            print(f"[button] GPIO unavailable ({e}); "
                  f"use .trigger() for testing")

    def _on_press(self):
        self._press_start = time.time()
        with self._lock:
            self._count += 1

    def _on_release(self):
        held = time.time() - self._press_start
        if held >= self.long_press_sec:
            with self._lock:
                self._count = 0
            self._fire(self.on_long)
            return
        if self._timer is None or not self._timer.is_alive():
            self._timer = threading.Timer(self.double_gap,
                                          self._resolve)
            self._timer.start()

    def _resolve(self):
        with self._lock:
            n, self._count = self._count, 0
        if   n == 1: self._fire(self.on_single)
        elif n == 2: self._fire(self.on_double)
        elif n >= 3: self._fire(self.on_triple)

    @staticmethod
    def _fire(cb):
        if cb:
            try:
                cb()
            except Exception as e:
                print(f"[button] callback error: {e}")

    def trigger(self, n=1):
        """Manual trigger for testing without GPIO."""
        with self._lock:
            self._count = n
        self._resolve()
