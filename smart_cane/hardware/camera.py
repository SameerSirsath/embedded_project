import numpy as np


class OpenCVCamera:
    def __init__(self, index=0, width=1280, height=720):
        import cv2
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {index} not available")

    def capture(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera capture failed")
        return frame

    def release(self):
        self.cap.release()


class PiCamera2Capture:
    def __init__(self, width=1280, height=720):
        from picamera2 import Picamera2
        self.picam = Picamera2()
        cfg = self.picam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.picam.configure(cfg)
        self.picam.start()

    def capture(self) -> np.ndarray:
        import cv2
        rgb = self.picam.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def release(self):
        self.picam.stop()
