"""Configuration for the standalone road-trajectory mapper.

Kept separate from the pipeline's root `config.py` so this tool can be run
and tuned on its own without touching turn detection.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# --- Input / output ------------------------------------------------------
# The mapper works on the already-split turn clips (one turn per file), not on
# the raw driving footage. Pass a folder to map every clip inside it.
DATASET_DIR = os.path.join(_ROOT, "dataset_turns_3class")
VIDEO_PATH = os.path.join(DATASET_DIR, "Sharp", "turn_0058.mp4")
OUTPUT_DIR = os.path.join(_HERE, "output")
SAVE_OUTPUT = True
SHOW_PLOT = True

# --- Pre-processing ------------------------------------------------------
# Dense Farneback flow at 1080p costs tens of ms per frame; 640x360 keeps the
# loop near real time and the trajectory shape is unaffected.
PROCESS_SIZE = (640, 360)
FRAME_STEP = 1  # analyse every Nth video frame (1 = every frame)

# Only the road surface may contribute to the flow average. Everything above
# ROI_Y[0] (the sky / distant hillside) is blanked out before the flow pass.
ROI_Y = (0.45, 0.85)
ROI_X = (0.25, 0.75)
MEAN_STRIDE = 2  # average every Nth ROI pixel (Farneback itself stays dense)

# --- Optical flow --------------------------------------------------------
# cv2.calcOpticalFlowFarneback(pyr_scale, levels, winsize, iterations,
#                              poly_n, poly_sigma, flags)
FARNEBACK = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=21,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)

# --- Pixels -> radians ---------------------------------------------------
# Yaw increment per analysed frame = YAW_SIGN * PIXELS_TO_RADIANS * OF_x.
PIXELS_TO_RADIANS = 0.003  # calibration: full lock-to-lock turn ~ 90 degrees
YAW_SIGN = -1  # -1 keeps left turns on the left; +1 if they come out mirrored

# --- Noise suppression ---------------------------------------------------
SMOOTHING_WINDOW = 1  # rolling average over the last N flow steps
DEADBAND_PX = 0.5  # |OF_x| below this is treated as perfectly straight

# --- Odometry (kinematic bicycle) ---------------------------------------
# Fallback when the speedometer cannot be read (and for the very first
# frames until the first reading arrives).
SPEED_KMH = 40.0
FALLBACK_STEP_M = 1.0  # metres per analysed step when SPEED_KMH is also None

# --- Speed telemetry -----------------------------------------------------
# Uses the pipeline's EasyOCR SpeedRecognizer (root speed_recognizer.py) to
# read the real speedometer value while mapping.
USE_RECOGNISED_SPEED = True
SPEED_SAMPLE_STEP = 10  # run OCR every N video frames
SPEED_AVERAGE_WINDOW = 5  # average the last N valid readings
SPEED_MIN_KMH = 1  # ignore zero / garbage reads
SPEED_MAX_KMH = 200  # ignore impossible reads

# --- Visualisation -------------------------------------------------------
SHOW_PREVIEW = True
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540
INSET_SIZE = (150, 260)  # portrait: lateral across, forward up
PLOT_FIGSIZE = (7, 11)  # portrait: the bike travels up the page
