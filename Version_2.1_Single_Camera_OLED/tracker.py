"""
tracker.py
Wraps `supervision`'s ByteTrack implementation to assign stable IDs to
each detected person across frames.
"""

import numpy as np
import supervision as sv

import config


class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=config.LOST_TRACK_BUFFER,
            minimum_matching_threshold=config.MINIMUM_MATCHING_THRESHOLD,
            frame_rate=config.TRACKER_FRAME_RATE,
        )

    def update(self, detections):
        """
        detections: list of [x1, y1, x2, y2, confidence]
        Returns a list of dicts: {track_id, x1, y1, x2, y2, confidence}
        """
        if len(detections) == 0:
            sv_detections = sv.Detections.empty()
        else:
            arr = np.array(detections, dtype=np.float32)
            sv_detections = sv.Detections(
                xyxy=arr[:, :4],
                confidence=arr[:, 4],
                class_id=np.zeros(len(arr), dtype=int),  # all "person"
            )

        tracked = self.tracker.update_with_detections(sv_detections)

        results = []
        if tracked.tracker_id is not None:
            for i in range(len(tracked)):
                x1, y1, x2, y2 = tracked.xyxy[i]
                results.append({
                    "track_id": int(tracked.tracker_id[i]),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "confidence": float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                })
        return results
