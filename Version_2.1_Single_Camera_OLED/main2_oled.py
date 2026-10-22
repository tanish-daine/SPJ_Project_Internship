"""
main2.py
Single-camera Re-Identification (Re-ID) extension of the Entry-Exit
Counting System, now with an always-on OLED as the PRIMARY output and the
Flask video dashboard as a SECONDARY, on-demand output.

Runs the SAME single-camera pipeline as main.py (Camera -> Detector ->
Tracker -> Counter -> Draw -> Stream -> Persist), plus:
  1. Re-ID (ReIdentity, below): a person keeps the SAME stable ID (1, 2,
     3, ...) through occlusion or briefly leaving and re-entering frame,
     instead of ByteTrack handing out a new, ever-climbing number.
  2. OLED (OledDisplay, below): Entry/Exit/Occupancy are pushed to a
     128x64 I2C SSD1306 OLED every loop iteration, UNCONDITIONALLY --
     this is the 24/7 always-on readout for unattended deployment.

WHY THE OLED IS "PRIMARY" AND THE DASHBOARD IS "SECONDARY"
------------------------------------------------------------
Flask's /video_feed route (stream.py, unmodified) only reads the latest
annotated frame from `broker` WHEN A BROWSER ASKS FOR IT. Nobody has to be
watching for the inference loop to keep running -- `broker.set_frame()` is
called every frame regardless of whether any browser is connected. That
was already true before this change.

The OLED update call below is placed BEFORE broker.set_frame() in the
loop, and is called every iteration with no "is anyone watching" check at
all -- it is the counting system's primary, always-on readout, meant to
run 24/7 whether or not anyone ever opens a browser. The Flask dashboard
stays exactly as before: optional, for someone to open occasionally to
visually confirm the OLED's numbers look correct, then close again.
Opening or closing the browser has zero effect on the OLED or on the
counting pipeline -- they are two independent consumers of the same
entry/exit/occupancy numbers.

If the OLED is missing, miswired, or hits an I2C error, OledDisplay
disables itself (after printing a warning) rather than crashing the
program -- a faulty display must never take down 24/7 counting.

LINE ORIENTATION (replaces the old SIDE/TOP camera-mode wording)
------------------------------------------------------------------
Set LINE_ORIENTATION below to choose the line instead -- BOTH orientations are still fully supported, just described by
what the line does, not by a camera-mounting label:

  LINE_ORIENTATION = "VERTICAL"    -> a vertical line; counts a person
                                       crossing left <-> right.
  LINE_ORIENTATION = "HORIZONTAL"  -> a horizontal line; counts a person
                                       crossing top <-> bottom.

VALIDATION NOTE (from project learnings): confirm Re-ID behaviour with
real-time physical walk-throughs, not pre-recorded or screen-played video
-- stock footage was previously found to distort ID assignment and inflate
ID numbers on this project.
"""

import time
import threading

import cv2
import numpy as np

import config
from camera import Camera
from detect import Detector
from tracker import Tracker
from counter import Counter
from persistence import Database
from stream import FrameBroker, create_app
from main import LatestFrame, draw_overlay, capture_thread_fn


# ---------------- Line orientation (see module docstring) ----------------
LINE_ORIENTATION = "HORIZONTAL"   # "VERTICAL" or "HORIZONTAL" -- edit this line only
_ORIENTATION_TO_COUNTER_MODE = {"VERTICAL": "SIDE", "HORIZONTAL": "TOP"}
COUNTER_MODE = _ORIENTATION_TO_COUNTER_MODE[LINE_ORIENTATION]

# ---------------- Re-ID tuning ----------------
ID_CONFIRM_FRAMES = 5        # frames a new local track must live before it gets a stable ID
                             # (lets the box fully enter the frame -> cleaner colour signature)
LOST_TIMEOUT_SEC = 2.0       # local track unseen this long -> treated as "gone from view"
HANDOFF_WINDOW_SEC = 8.0     # how long a lost ID stays claimable by a reappearing person
MATCH_THRESHOLD = 0.45       # Bhattacharyya distance: 0 = identical, 1 = totally different.
                             # Lower = stricter (fewer wrong re-matches, more ID resets).
                             # Raise (0.5-0.6) if a returning person isn't getting their old ID back.

# ---------------- OLED tuning ----------------
OLED_I2C_PORT = 1            # Raspberry Pi's default I2C bus (the SDA/SCL pins)
OLED_I2C_ADDRESS = 0x3C      # standard SSD1306 address; some boards use 0x3D -- check `i2cdetect -y 1`
OLED_REFRESH_SEC = 0.5       # redraw at most twice a second -- a full-screen write has no benefit
                             # happening faster than a human can read it


class ReIdentity:
    """
    Turns this camera's raw, ever-incrementing ByteTrack IDs into STABLE
    IDs that start at 1 and persist across occlusion / leaving-and-
    re-entering the frame.

    How a person is re-recognised after their track is lost (single camera,
    no calibration needed):
      - Signature: HSV colour histogram of the person's torso region.
      - Lifecycle: when a local track is lost, its (stable ID, signature)
        is parked in a "lost pool" for HANDOFF_WINDOW_SEC. A brand-new
        local track is compared against that pool; the best match under
        MATCH_THRESHOLD inherits the old stable ID, otherwise a new ID
        is issued.

    Limits: two people in near-identical clothing can swap IDs if one
    leaves as the other arrives. Lighting changes across the frame can
    also weaken the match -- raise MATCH_THRESHOLD if returning people
    aren't getting their old ID back, lower it if two different people
    are getting merged into one ID.
    """

    def __init__(self):
        self.next_id = 1                 # stable IDs start at 1
        self.local_to_stable = {}        # local_track_id -> stable_id
        self.last_seen = {}              # local_track_id -> timestamp (confirmed tracks)
        self.pending = {}                # local_track_id -> {"frames", "sig", "last"}
        self.signatures = {}             # stable_id -> colour signature
        self.lost_pool = []              # [{"id", "sig", "time"}]

    # ---------- signature ----------
    @staticmethod
    def _signature(frame, obj):
        h, w = frame.shape[:2]
        x1 = int(max(0, obj["x1"]))
        y1 = int(max(0, obj["y1"]))
        x2 = int(min(w, obj["x2"]))
        y2 = int(min(h, obj["y2"]))
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 16:
            return None

        # Central torso strip: skip the edges (background) and the head/legs.
        cx1 = x1 + int(0.2 * bw)
        cx2 = x2 - int(0.2 * bw)
        cy1 = y1 + int(0.15 * bh)
        cy2 = y1 + int(0.65 * bh)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4],
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist.flatten().astype(np.float32)

    # ---------- matching ----------
    def _match_or_new(self, sig, now):
        best_idx, best_dist = None, MATCH_THRESHOLD
        if sig is not None:
            for i, entry in enumerate(self.lost_pool):
                if now - entry["time"] > HANDOFF_WINDOW_SEC:
                    continue
                dist = cv2.compareHist(sig, entry["sig"], cv2.HISTCMP_BHATTACHARYYA)
                if dist < best_dist:
                    best_idx, best_dist = i, dist

        if best_idx is not None:
            return self.lost_pool.pop(best_idx)["id"]

        sid = self.next_id
        self.next_id += 1
        return sid

    # ---------- housekeeping ----------
    def _expire(self, now):
        for local_id in list(self.local_to_stable.keys()):
            if now - self.last_seen[local_id] > LOST_TIMEOUT_SEC:
                sid = self.local_to_stable.pop(local_id)
                seen = self.last_seen.pop(local_id)
                sig = self.signatures.get(sid)
                if sig is not None:
                    self.lost_pool.append({"id": sid, "sig": sig, "time": seen})

        for local_id in list(self.pending.keys()):
            if now - self.pending[local_id]["last"] > LOST_TIMEOUT_SEC:
                del self.pending[local_id]

        self.lost_pool = [e for e in self.lost_pool
                          if now - e["time"] <= HANDOFF_WINDOW_SEC]

        keep = set(self.local_to_stable.values()) | {e["id"] for e in self.lost_pool}
        self.signatures = {s: sig for s, sig in self.signatures.items() if s in keep}

    # ---------- public API ----------
    def resolve(self, tracks, frame, now):
        """
        tracks: list of dicts from tracker.py (local track_id, x1, y1, x2, y2, ...)
        Returns a new list of the same dicts with track_id replaced by the
        STABLE id. Tracks still inside their ID_CONFIRM_FRAMES warm-up are
        left out until confirmed, so the overlay never flashes a "wrong"
        ID that later gets corrected.
        """
        resolved = []
        for obj in tracks:
            local_id = obj["track_id"]
            sid = self.local_to_stable.get(local_id)

            if sid is None:
                p = self.pending.setdefault(local_id, {"frames": 0, "sig": None, "last": now})
                p["frames"] += 1
                p["last"] = now

                sig = self._signature(frame, obj)
                if sig is not None:
                    if p["sig"] is None:
                        p["sig"] = sig
                    else:
                        p["sig"] = (0.7 * p["sig"] + 0.3 * sig).astype(np.float32)

                if p["frames"] < ID_CONFIRM_FRAMES:
                    continue

                sid = self._match_or_new(p["sig"], now)
                self.local_to_stable[local_id] = sid
                if p["sig"] is not None:
                    self.signatures[sid] = p["sig"]
                del self.pending[local_id]

            self.last_seen[local_id] = now
            new_obj = dict(obj)
            new_obj["track_id"] = sid
            resolved.append(new_obj)

        self._expire(now)
        return resolved


class OledDisplay:
    """
    Drives a 128x64 SSD1306 I2C OLED to show live Entry / Exit / Occupancy.
    This is the PRIMARY, always-on output for 24/7 unattended monitoring --
    update() is called every loop iteration in inference_loop2, completely
    independent of whether anyone has the Flask video dashboard open.
    The dashboard stays a SECONDARY, on-demand view: someone can open
    /video_feed to visually confirm the OLED's counts look right, then
    close it -- that never pauses, slows, or otherwise affects the OLED
    or the counting pipeline, because they both just read the same
    entry/exit/occupancy numbers, independently, every frame.

    If the OLED is missing, miswired, or the I2C bus glitches, this class
    disables itself after printing a warning instead of raising -- a
    display fault must never be able to stop 24/7 counting.
    """

    def __init__(self):
        self.enabled = True
        self.last_drawn = None     # last (entry, exit_, occupancy) actually pushed to the panel
        self.last_update = 0.0
        try:
            # Imported here, not at module level, so a missing OLED library
            # on a machine that isn't the deployed Pi (e.g. a dev laptop)
            # doesn't block importing this file at all.
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306
            from PIL import ImageFont

            serial = i2c(port=OLED_I2C_PORT, address=OLED_I2C_ADDRESS)
            self.device = ssd1306(serial, width=128, height=64)
            self.font_big = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
            )
            self.font_small = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
            )
        except Exception as e:
            print(f"[OLED] disabled -- init failed: {e}")
            self.enabled = False

    def update(self, entry, exit_, occupancy, now):
        if not self.enabled:
            return

        # Throttle: a full-screen I2C redraw has no benefit happening
        # faster than OLED_REFRESH_SEC -- the numbers don't change that fast.
        if now - self.last_update < OLED_REFRESH_SEC:
            return

        values = (entry, exit_, occupancy)
        if values == self.last_drawn:
            self.last_update = now
            return

        try:
            from luma.core.render import canvas
            with canvas(self.device) as draw:
                draw.text((0, 0), "OCCUPANCY", font=self.font_small, fill="white")
                draw.text((0, 14), str(occupancy), font=self.font_big, fill="white")
                draw.text((0, 48), f"Entry:{entry}", font=self.font_small, fill="white")
                draw.text((64, 48), f"Exit:{exit_}", font=self.font_small, fill="white")
            self.last_drawn = values
            self.last_update = now
        except Exception as e:
            # A transient I2C error must never be able to take down the
            # counting loop -- skip this refresh and try again next time.
            print(f"[OLED] write failed, skipping this refresh: {e}")


def inference_loop2(latest, broker, db, reid, oled):
    detector = Detector()
    tracker = Tracker()
    counter = Counter(mode=COUNTER_MODE, line_position=config.COUNT_LINE_POSITION)

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
        # else: reuse last_detections -- same coasting behavior as main.py

        now = time.time()

        tracks = tracker.update(last_detections)
        # Swap ByteTrack's raw, ever-incrementing local IDs for STABLE IDs
        # that persist through occlusion / leaving-and-re-entering the frame.
        tracks = reid.resolve(tracks, frame, now)

        entry, exit_, occupancy = counter.update(tracks, frame.shape[1], frame.shape[0])

        # PRIMARY output: push to the OLED first, every iteration, no matter
        # whether a browser is connected to the Flask dashboard or not.
        oled.update(entry, exit_, occupancy, now)

        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = (0.9 * fps) + (0.1 * (1.0 / dt))

        # SECONDARY output: this only matters if/when someone opens a
        # browser -- broker.set_frame() is cheap to call either way, but
        # nothing downstream of it can affect the OLED update above.
        annotated = draw_overlay(frame.copy(), tracks, counter, fps)
        broker.set_frame(annotated, entry, exit_, occupancy, fps)

        if now - last_db_write >= config.DB_FLUSH_INTERVAL_SEC:
            db.log(entry, exit_, occupancy)
            last_db_write = now


def main():
    camera = Camera()
    latest = LatestFrame()
    broker = FrameBroker()
    db = Database()
    reid = ReIdentity()
    oled = OledDisplay()

    threading.Thread(target=capture_thread_fn, args=(camera, latest), daemon=True).start()
    threading.Thread(target=inference_loop2, args=(latest, broker, db, reid, oled), daemon=True).start()

    app = create_app(broker)
    app.run(host=config.STREAM_HOST, port=config.STREAM_PORT, threaded=True)


if __name__ == "__main__":
    main()
