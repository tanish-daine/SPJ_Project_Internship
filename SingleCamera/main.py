"""
main.py
Capture runs in its own thread and always holds only the latest frame.
Inference runs in a second thread, always grabbing whatever's newest —
this prevents the growing lag you get when frames queue up faster than
the detector can process them.
"""

import time
import threading

import cv2

import config
from camera import Camera
from detect import Detector
from tracker import Tracker
from counter import Counter
from persistence import Database
from stream import FrameBroker, create_app


class LatestFrame:
    """Holds only the single most recent raw frame — old ones are discarded, not queued."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def set(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


def draw_overlay(frame, tracks, counter: Counter, fps):
    line_pos = counter.line_pixel_for_drawing(frame.shape[1], frame.shape[0])

    if counter.mode == "SIDE":
        cv2.line(frame, (line_pos, 0), (line_pos, frame.shape[0]), (0, 0, 255), 2)
    else:
        cv2.line(frame, (0, line_pos), (frame.shape[1], line_pos), (0, 0, 255), 2)

    for obj in tracks:
        x1, y1, x2, y2 = map(int, (obj["x1"], obj["y1"], obj["x2"], obj["y2"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {obj["track_id"]}', (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Entry: {counter.entry_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Exit: {counter.exit_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Occupancy: {counter.occupancy}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def capture_thread_fn(camera, latest):
    while True:
        latest.set(camera.get_frame())


def inference_loop(latest, broker):
    detector = Detector()
    tracker = Tracker()
    counter = Counter(mode=config.CAMERA_MODE, line_position=config.COUNT_LINE_POSITION)
    db = Database()

    prev_time = time.time()
    fps = 0.0
    last_db_write = time.time()
    frame_counter = 0
    last_detections = []

    while True:
        frame = latest.get()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_counter += 1
        if frame_counter % config.DETECT_EVERY_N_FRAMES == 0:
            last_detections = detector.detect(frame)
        # else: reuse last_detections — ByteTrack's own motion model
        # covers the gap, so counting accuracy stays intact.

        tracks = tracker.update(last_detections)
        entry, exit_, occupancy = counter.update(tracks, frame.shape[1], frame.shape[0])

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = (0.9 * fps) + (0.1 * (1.0 / dt))

        annotated = draw_overlay(frame.copy(), tracks, counter, fps)
        broker.set_frame(annotated, entry, exit_, occupancy, fps)

        if now - last_db_write >= config.DB_FLUSH_INTERVAL_SEC:
            db.log(entry, exit_, occupancy)
            last_db_write = now


def main():
    camera = Camera()
    latest = LatestFrame()
    broker = FrameBroker()
    app = create_app(broker)

    threading.Thread(target=capture_thread_fn, args=(camera, latest), daemon=True).start()
    threading.Thread(target=inference_loop, args=(latest, broker), daemon=True).start()

    app.run(host=config.STREAM_HOST, port=config.STREAM_PORT, threaded=True)


if __name__ == "__main__":
    main()
