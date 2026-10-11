"""
stream.py
Flask app that serves the live annotated MJPEG stream + a simple dashboard page.

main.py continuously pushes the latest annotated JPEG bytes into a
FrameBroker; this module only reads from it and serves it over HTTP.
Kept separate from main.py so the web layer can be swapped later
(e.g. for Gunicorn + systemd) without touching the vision pipeline.
"""

import threading

import cv2
from flask import Flask, Response, render_template

import config


class FrameBroker:
    """Thread-safe holder for the most recent annotated frame + live stats."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg_bytes = None
        self.stats = {"entry": 0, "exit": 0, "occupancy": 0, "fps": 0.0}

    def set_frame(self, frame, entry, exit_, occupancy, fps):
        display_frame = cv2.resize(
            frame, (config.STREAM_DISPLAY_WIDTH, config.STREAM_DISPLAY_HEIGHT)
        )
        ok, buffer = cv2.imencode(
            ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]
        )
        if not ok:
            return
        with self._lock:
            self._jpeg_bytes = buffer.tobytes()
            self.stats = {"entry": entry, "exit": exit_, "occupancy": occupancy, "fps": fps}

    def get_jpeg(self):
        with self._lock:
            return self._jpeg_bytes


def create_app(broker: FrameBroker) -> Flask:
    app = Flask(__name__)

    def generate():
        while True:
            jpeg = broker.get_jpeg()
            if jpeg is None:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stats")
    def stats():
        return broker.stats

    return app
