"""End-to-end OCR+TTS test without GPIO."""
import time

from smart_cane.hardware.camera import OpenCVCamera
from smart_cane.services.audio import ReadingTTS
from smart_cane.services.ocr_reader import PageReader


cam    = OpenCVCamera(0, 1280, 720)
tts    = ReadingTTS(rate=170)
reader = PageReader(capture_fn=cam.capture, engine="rapidocr",
                    mode="book")

print("Hold a page in front of the camera. Reading in 3 s...")
time.sleep(3)

img  = reader.grab()
text = reader.extract_text(img)
print(f"\n--- Extracted ({len(text)} chars) ---\n{text[:500]}\n")

if text.strip():
    tts.speak(text)
