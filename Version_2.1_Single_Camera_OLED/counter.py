"""
counter.py
Hysteresis-based entry/exit counting logic.

Instead of counting the instant a track crosses the line, we require the
track's centroid to remain on the NEW side of the line for
HYSTERESIS_FRAMES consecutive frames before the crossing is confirmed.
This suppresses false counts caused by detection jitter around the line.
"""

import config


class Counter:
    def __init__(self, mode=None, line_position=None):
        self.mode = (mode or config.CAMERA_MODE).upper()
        self.line_ratio = line_position if line_position is not None else config.COUNT_LINE_POSITION

        self.entry_count = 0
        self.exit_count = 0
        self.occupancy = 0

        self.tracks = {}

    def _line_pixel(self, frame_w, frame_h):
        if self.mode == "SIDE":
            return int(self.line_ratio * frame_w)
        return int(self.line_ratio * frame_h)   # TOP mode -> horizontal line

    def update(self, tracked_objects, frame_w, frame_h):
        """
        tracked_objects: list of dicts with track_id, x1, y1, x2, y2 (from tracker.py)
        Returns (entry_count, exit_count, occupancy) after processing this frame.
        """
        line = self._line_pixel(frame_w, frame_h)
        active_ids = set()

        for obj in tracked_objects:
            tid = obj["track_id"]
            active_ids.add(tid)

            cx = (obj["x1"] + obj["x2"]) / 2
            cy = (obj["y1"] + obj["y2"]) / 2

            position_value = cx if self.mode == "SIDE" else cy
            side = "positive" if position_value >= line else "negative"

            if tid not in self.tracks:
                self.tracks[tid] = {
                    "side": side,
                    "pending_side": side,
                    "pending_count": 0,
                    "counted": False,
                    "history_len": 1,
                }
                continue

            state = self.tracks[tid]
            state["history_len"] += 1

            if side == state["side"]:
                state["pending_side"] = side
                state["pending_count"] = 0
                continue

            if state["pending_side"] == side:
                state["pending_count"] += 1
            else:
                state["pending_side"] = side
                state["pending_count"] = 1

            if (
                state["pending_count"] >= config.HYSTERESIS_FRAMES
                and not state["counted"]
                and state["history_len"] >= config.MIN_TRACK_HISTORY
            ):
                if state["side"] == "negative" and side == "positive":
                    self.entry_count += 1
                elif state["side"] == "positive" and side == "negative":
                    self.exit_count += 1

                state["counted"] = False   # reset so a later crossing still counts
                state["side"] = side
                state["pending_count"] = 0

        stale_ids = [tid for tid in self.tracks if tid not in active_ids]
        for tid in stale_ids:
            del self.tracks[tid]

        self.occupancy = max(0, self.entry_count - self.exit_count)
        self.live_count = len(active_ids)
        return self.entry_count, self.exit_count, self.occupancy

    def line_pixel_for_drawing(self, frame_w, frame_h):
        return self._line_pixel(frame_w, frame_h)
