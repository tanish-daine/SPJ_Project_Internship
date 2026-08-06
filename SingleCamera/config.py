"""
config.py
All tunable parameters for the Entry-Exit Counting System V2.
Change values here — never inside the other modules — to redeploy at a
new site or re-tune accuracy.
"""

# ---------------- Camera ----------------
CAMERA_MODE = "TOP"          # "SIDE" or "TOP"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 20

# ---------------- Detection ----------------
MODEL_PATH = "models/yolo11n.onnx"
MODEL_INPUT = 640             # square input size the model was exported with
CONFIDENCE_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.45
PERSON_CLASS_ID = 0           # COCO class index for "person"

# ---------------- Tracking (ByteTrack via `supervision`) ----------------
TRACK_ACTIVATION_THRESHOLD = 0.5   # min confidence to start/keep a track
LOST_TRACK_BUFFER = 30             # frames to keep a lost track alive
MINIMUM_MATCHING_THRESHOLD = 0.8   # IoU threshold for match
TRACKER_FRAME_RATE = 20            # should roughly match TARGET_FPS

# ---------------- Counting ----------------
# Fraction of the frame where the virtual counting line sits.
# SIDE mode -> fraction of FRAME_WIDTH (vertical line)
# TOP mode  -> fraction of FRAME_HEIGHT (horizontal line)
COUNT_LINE_POSITION = 0.50
HYSTERESIS_FRAMES = 3          # consecutive frames required to confirm a crossing
MIN_TRACK_HISTORY = 3          # ignore very short-lived tracks (likely noise)

# ---------------- Streaming ----------------
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 5000
JPEG_QUALITY = 80              # 0-100, lower = faster encode / smaller frame

# ---------------- Storage ----------------
DB_PATH = "database/occupancy.db"
DB_FLUSH_INTERVAL_SEC = 5

# ---------------- Performance tuning ----------------

INTRA_OP_THREADS = 4            # Pi 5 has 4 cores — let ONNX Runtime use all of them
DETECT_EVERY_N_FRAMES = 2       # run full detection every Nth frame; tracker coasts in between
STREAM_DISPLAY_WIDTH = 854      # resize before JPEG encode — detection still uses full MODEL_INPUT
STREAM_DISPLAY_HEIGHT = 480
