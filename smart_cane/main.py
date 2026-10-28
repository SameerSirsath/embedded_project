from .config import Config
from .core.mode_manager import ModeManager
from .hardware.button import ButtonController
from .hardware.camera import OpenCVCamera, PiCamera2Capture
from .services.audio import ReadingTTS
from .services.bookmarks import BookmarkStore
from .services.ocr_reader import PageReader


def make_camera():
    try:
        return PiCamera2Capture(Config.CAMERA_WIDTH,
                                Config.CAMERA_HEIGHT)
    except Exception:
        return OpenCVCamera(Config.CAMERA_INDEX,
                            Config.CAMERA_WIDTH,
                            Config.CAMERA_HEIGHT)


def init_reader_subsystem(detector_controller=None, capture_fn=None):
    """
    Call once from smart_ai_cane.py at startup.
    Returns (mode_manager, tts).
    """
    tts = ReadingTTS(rate=Config.TTS_RATE, voice=Config.TTS_VOICE)

    if capture_fn is None:
        try:
            cam = make_camera()
            capture_fn = cam.capture
        except Exception as e:
            print(f"[reader] camera init note: {e}")
            capture_fn = lambda: None

    bookmarks       = BookmarkStore(Config.BOOKMARKS_PATH)
    reader          = PageReader(
        capture_fn   = capture_fn,
        engine       = Config.OCR_ENGINE,
        mode         = Config.OCR_MODE,
        burst_frames = Config.OCR_BURST_FRAMES,
        burst_delay  = Config.OCR_BURST_DELAY,
    )
    reader.bookmarks = bookmarks

    button = ButtonController(
        pin            = Config.BUTTON_PIN,
        long_press_sec = Config.BUTTON_LONG_PRESS_SEC,
        double_gap     = Config.BUTTON_DOUBLE_GAP,
    )

    manager = ModeManager(reader, tts, button, detector_controller)
    manager.start()
    tts.speak("Smart cane ready.")
    return manager, tts
