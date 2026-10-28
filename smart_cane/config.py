import os


class Config:
    # --- Camera -------------------------------------------------
    CAMERA_INDEX   = int(os.getenv("SC_CAMERA_INDEX", "0"))
    CAMERA_WIDTH   = 1280
    CAMERA_HEIGHT  = 720

    # --- OCR ----------------------------------------------------
    OCR_ENGINE      = os.getenv("SC_OCR_ENGINE", "rapidocr")  # rapidocr | tesseract
    OCR_MODE        = os.getenv("SC_OCR_MODE", "book")        # book | screen
    OCR_BURST_FRAMES = 3
    OCR_BURST_DELAY  = 0.15

    # --- TTS ----------------------------------------------------
    TTS_RATE  = 170
    TTS_VOICE = None   # substring of voice name, e.g. "english"

    # --- Button (BCM pin number) --------------------------------
    BUTTON_PIN            = int(os.getenv("SC_BUTTON_PIN", "17"))
    BUTTON_LONG_PRESS_SEC = 1.5
    BUTTON_DOUBLE_GAP     = 0.4

    # --- Bookmarks ----------------------------------------------
    BOOKMARKS_PATH = os.getenv("SC_BOOKMARKS_PATH",
                               "~/.smart_cane_bookmarks.json")

    # --- FPS modes ----------------------------------------------
    NAV_FPS     = 15
    READING_FPS = 2
