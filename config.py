"""Central configuration for the turn-detection dataset pipeline."""

# Input source: local file path OR YouTube URL
INPUT_SOURCE = "./data_1.mp4"

# Output locations
DATASET_DIR = "dataset_turns_3class"
EXCEL_FILE = "road_turn_dataset.xlsx"
CLASSES = ("Minor", "Major", "Sharp")

# Display
SHOW_PREVIEW = True  # Set False on a headless server
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# Optical flow sampling
FRAME_STEP = 3  # Compute flow every 3 frames (~0.1s step)

# Turn activation
TURN_THRESHOLD = 0.25  # Lowered to catch light/minor road bends
MIN_TURN_DURATION = 6  # Minimum steps to confirm a turn

# Clip duration (20s total: 10s pre + 10s post)
PRE_TURN_SECONDS = 10
POST_TURN_SECONDS = 10

# Noise suppression
CAMERA_CUT_DISCONTINUITY = 25.0
SMOOTHING_WINDOW = 5

# Threading
FRAME_QUEUE_SIZE = 128  # Bounded queue between reader thread and processor
CLIP_WRITER_WORKERS = 2  # Concurrent background clip-encoding threads
READER_QUEUE_TIMEOUT = 5.0  # Seconds to wait for a frame before giving up

# Classification boundaries (accumulated horizontal deflection)
DEFLECTION_DISCARD = 10.0
DEFLECTION_MINOR_MAX = 30.0
DEFLECTION_MAJOR_MAX = 60.0

# ROI for optical flow, as fractions of frame height/width
ROI_Y = (0.45, 0.85)
ROI_X = (0.25, 0.75)

# Horizontal-dominance ratio for the turn-vector validity check
RATIO_DOMINANCE = 1.05
