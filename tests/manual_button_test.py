import time

from smart_cane.hardware.button import ButtonController
from smart_cane.config import Config


def say(msg):
    print(f"→ {msg}")


ButtonController(
    pin       = Config.BUTTON_PIN,
    on_single = lambda: say("SINGLE"),
    on_double = lambda: say("DOUBLE"),
    on_triple = lambda: say("TRIPLE"),
    on_long   = lambda: say("LONG"),
)
print(f"Listening on GPIO {Config.BUTTON_PIN}. Ctrl+C to quit.")
while True:
    time.sleep(1)
