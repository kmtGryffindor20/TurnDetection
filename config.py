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

# --- Speedometer OCR (EasyOCR, GPU) ---
# On turn completion the lower part of the frame is OCR'd. If any numeric text
# is found, the largest numeric token is stored as the clip's speed and the turn
# is saved; otherwise the turn is discarded. There is deliberately no
# plausibility (0-199) or confidence check -- EasyOCR's learned detector already
# returns nothing when there is no text, which is what keeps scenery from being
# read as a speed.
OCR_ENABLED = True
OCR_ROI_Y = (0.50, 0.88)  # band inside the lower half of the frame (console)
OCR_ROI_X = (0.15, 0.72)
OCR_DIGIT_ALLOWLIST = "0123456789"
OCR_DETECT_MAG_RATIO = 3.0  # internal upscale so the small glyphs are detected
OCR_LOW_TEXT = 0.3  # EasyOCR detector threshold; lower catches fainter text
# Total frames OCR'd per turn: the completion frame + recent pre-turn frames.
OCR_GATE_SAMPLES = 5

# EasyOCR engine settings
EASYOCR_LANGS = ("en",)
EASYOCR_GPU = True  # requires a CUDA build of torch (see README)
EASYOCR_MODEL_DIR = "vendor/easyocr_models"  # git-ignored; keeps models local

# Live speedometer OCR for the HUD: in addition to the turn-completion gate,
# OCR runs every OCR_LIVE_INTERVAL_SECONDS while driving so the detected box and
# speed track the readout in real time. EasyOCR on GPU is fast enough (~0.2s per
# pass); on CPU raise this interval or set OCR_LIVE_ENABLED = False.
OCR_LIVE_ENABLED = True
OCR_LIVE_INTERVAL_SECONDS = 0.5
