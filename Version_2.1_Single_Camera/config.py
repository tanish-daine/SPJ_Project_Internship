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

# ---------------- Dual-Camera Fusion (Task 1) ----------------
# values after the physical measurement step (Stage below).
# Each entry: (pixel_x, pixel_y, world_x_cm, world_y_cm)
# Minimum 4 points per camera; 6-8 spread across the frame is more robust.
# IMPORTANT: both cameras' world coordinates must share the SAME physical
# origin and axes -- measure everything from one fixed room corner.
CAM1_CALIB_POINTS = [
    # (px, py, wx_cm, wy_cm)  -- REPLACE with your real measurements
    (100, 600,   0,   0),
    (1180, 600, 400,   0),
    (100, 100,   0, 300),
    (1180, 100, 400, 300),
]

CAM2_CALIB_POINTS = [
    # (px, py, wx_cm, wy_cm)  -- REPLACE with your real measurements
    (100, 600, 400,   0),
    (1180, 600, 800,   0),
    (100, 100, 400, 300),
    (1180, 100, 800, 300),
]

# How close two projected world points (cm) must be to be treated as the
# SAME physical person and merged, instead of counted twice at the seam.
DEDUP_RADIUS_CM = 60.0

# ONE corridor boundary line in real-world cm, used for entry/exit tally.
# Axis = whichever world axis runs ALONG the direction people walk.
GLOBAL_LINE_AXIS = "x"           # "x" or "y" -- match your calib layout
ENTRY_LINE_POSITION_CM = 50.0   # place near CAM1's outer/doorway edge
EXIT_LINE_POSITION_CM = 750.0     # place near CAM2's outer/doorway edge -- MUST be > ENTRY_LINE_POSITION_CM
GLOBAL_HYSTERESIS_FRAMES = 5
GLOBAL_MIN_TRACK_HISTORY = 3
