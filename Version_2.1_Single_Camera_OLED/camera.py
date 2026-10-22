"""
camera.py
Captures stable frames from the Raspberry Pi Camera Module 3 using Picamera2.
"""

import numpy as np
from picamera2 import Picamera2

import config


class Camera:
    def __init__(self):
        self.picam2 = Picamera2()

        video_config = self.picam2.create_video_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": config.TARGET_FPS},
        )
        self.picam2.configure(video_config)

        try:
            self.picam2.set_controls({"AfMode": 2, "AfSpeed": 1})  # 2 = Continuous
        except Exception:
            # Older libcamera builds may not expose these controls; safe to ignore.
            pass

        self.picam2.start()

    def get_frame(self):
        """
        Returns a BGR OpenCV frame of size (FRAME_WIDTH, FRAME_HEIGHT).
        Picamera2's "RGB888" format is already BGR byte order — no conversion needed.
        """
        frame = self.picam2.capture_array()
        return np.ascontiguousarray(frame[:, :, :3])

    def release(self):
        self.picam2.stop()
