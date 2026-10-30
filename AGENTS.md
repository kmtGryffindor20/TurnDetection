# TurnDetection — Agent Quick Reference

## Project Overview
Pipeline that detects vehicle turns from dashcam video (local file or YouTube URL) using dense optical flow (Farneback), exports 20-second clips (10s pre + 10s post) into a 3-class dataset (`Minor`/`Major`/`Sharp`), and writes an Excel catalog.

## Entry Point
```
python main.py
```
Edit `INPUT_SOURCE` in `config.py` (local path or YouTube URL). Set `SHOW_PREVIEW = False` for headless servers.

Batch mode (directory of group folders, sequential, tqdm progress):
```
python pipeline.py <root_dir> --output batch_output --no-preview
```
Each video gets its own `dataset_turns_3class/` + `road_turn_dataset.xlsx`
under the output dir; a combined catalog is written at the end. `config.py`
values are overridden per video and restored afterwards. Flags:
`--skip-existing`, `--dry-run`, `--no-preview`, `--quiet`, `--transcode`
(AV1 → cached H.264 via system ffmpeg: `<output>/.transcode_cache/`).

## Dependencies
```
pip install -r requirements.txt          # desktop (OpenCV GUI preview)
pip install -r requirements-server.txt   # headless VM: opencv-python-headless
```
For GPU speed recognition install a CUDA build of `torch`; with `SPEED_GPU = False` (or no CUDA) EasyOCR still works on CPU, just slower. On a headless server also install torch/torchvision first (CUDA or CPU index), set `SPEED_GPU` accordingly, and run with `--no-preview`. EasyOCR models live in git-ignored `vendor/easyocr/` (auto-downloaded on first use).

## Architecture (Threading Model)
| Component | Thread | Responsibility |
|-----------|--------|----------------|
| `FrameReader` | Background (daemon) | Reads/decodes frames into bounded queue; drops oldest frame when full to stay near-real-time |
| Main thread | Main | Pulls frames, runs `TurnDetector.update()`, manages rolling buffer, drives OpenCV preview (`imshow`/`waitKey` must stay on main thread) |
| `DatasetWriter` | ThreadPoolExecutor (`CLIP_WRITER_WORKERS=2`) | Encodes clips to disk in background; maintains thread-safe catalog flushed to Excel on shutdown |

## Key Files
- `config.py` — All tunable constants (thresholds, durations, paths, ROI, threading)
- `stream_utils.py` — `resolve_video_source()` (YouTube → direct URL via yt-dlp, excludes AV1), `FrameReader` class
- `optical_flow.py` — `TurnDetector` state machine; call `sync_gray(frame)` during post-roll capture to keep `prev_gray` current without full flow computation
- `classification.py` — `classify_turn(deflection)` → `None`/`Minor`/`Major`/`Sharp`
- `speed_recognizer.py` — `SpeedRecognizer.read(frame)` → `(speed_kmh|None, bbox|None)` via EasyOCR in the lower-middle ROI
- `dataset_writer.py` — `DatasetWriter` background encoding + Excel catalog
- `visualization.py` — HUD overlay + `PreviewWindow` (main-thread only)
- `main.py` — Wires everything together (single video)
- `pipeline.py` — Batch runner: directory of group folders, sequential; outer `Videos: n/N` bar + inner per-video frame bar (via `process_video_pipeline(progress_callback=...)`)

## Critical Design Notes
1. **Post-turn capture uses `sync_gray()`** — While capturing post-turn frames, `TurnDetector.sync_gray(frame)` advances the reference frame without running optical flow. This prevents the next `update()` from diffing frames seconds apart (which would trigger false camera-cut resets).
2. **Frame drop policy** — `FrameReader` drops oldest queued frame when full. This keeps pipeline current; it does NOT guarantee every frame is processed.
3. **OpenCV HighGUI** — Preview window (`cv2.imshow`/`waitKey`) MUST run on main thread. Never call from worker threads.
4. **YouTube codec selection** — yt-dlp format string prefers H.264 (avc1) / VP9, explicitly excludes AV1 (av01) which OpenCV often cannot decode.

## Config Highlights (`config.py`)
| Constant | Default | Notes |
|----------|---------|-------|
| `INPUT_SOURCE` | YouTube URL | Change to local path or different URL |
| `FRAME_STEP` | 3 | Compute flow every N frames (~0.1s at 30fps) |
| `TURN_THRESHOLD` | 0.25 | Horizontal flow magnitude to trigger turn |
| `MIN_TURN_DURATION` | 6 | Minimum flow steps to confirm a turn |
| `PRE_TURN_SECONDS` / `POST_TURN_SECONDS` | 10 / 10 | Clip = 20s total |
| `CAMERA_CUT_DISCONTINUITY` | 25.0 | Flow magnitude jump treated as camera cut (resets state) |
| `DEFLECTION_DISCARD` / `MINOR_MAX` / `MAJOR_MAX` | 10 / 30 / 60 | Classification boundaries (abs accumulated dx) |
| `ROI_Y` / `ROI_X` | (0.45, 0.85) / (0.25, 0.75) | Optical flow region of interest (fraction of frame) |
| `CLIP_WRITER_WORKERS` | 2 | Background encoding threads |
| `SPEED_ENABLED` | True | Gate: only save turns with a recognised speed |
| `SPEED_LIMIT_KMH` | 200.0 | Readings above this are rejected as noise |
| `SPEED_SAMPLE_STEP` | 6 | Recognise the readout every N frames |
| `SPEED_ROI_Y` / `SPEED_ROI_X` | (0.60, 1.00) / (0.25, 0.75) | Speedometer ROI (fraction of frame) |
| `VERBOSE` | True | Per-turn/per-clip internal logs; `pipeline.py --quiet` forces False |


## Output Structure
```
dataset_turns_3class/
  Minor/
  Major/
  Sharp/
road_turn_dataset.xlsx  # Catalog with Turn ID, Class, Deflection Value, Average Speed (km/h), Saved Clip Address
```

## Road Trajectory Tool (`road_trajectory/`)
Standalone tool (shares the venv and `speed_recognizer.SpeedRecognizer`) that
maps an already-split turn clip to a 2D top-down path: resize → grayscale →
blank above horizon → Farneback flow → mean horizontal displacement → rolling
average + deadband → `YAW_SIGN * PIXELS_TO_RADIANS` → kinematic bicycle
odometry, with the speedometer read by EasyOCR every `SPEED_SAMPLE_STEP`
frames.
```
python road_trajectory/main.py [clip-or-folder]   # default: one dataset clip
```
Classes: `FlowYawEstimator`, `KinematicBicycle`, `RoadTrajectoryMapper`
(`trajectory_mapper.py`), `SpeedSampler` (`main.py`); plotting/preview in
`trajectory_visualization.py`; tunables in `trajectory_config.py`.
Works on `dataset_turns_3class/` clips only (a folder arg maps them all).
Writes `road_trajectory/output/<Class>_<clip>.{png,csv}`; plots are vertical
(+Y forward = up, +X = right, positive heading = left turn). Key knobs:
`PIXELS_TO_RADIANS` (calibration), `YAW_SIGN` (left/right mirror). `flow.py` is
the original scratch script.

## Common Tasks
- **Change input source**: Edit `config.INPUT_SOURCE`
- **Run headless**: Set `config.SHOW_PREVIEW = False`
- **Adjust sensitivity**: Modify `TURN_THRESHOLD`, `MIN_TURN_DURATION`, classification boundaries
- **Change clip length**: Modify `PRE_TURN_SECONDS` / `POST_TURN_SECONDS`
- **Tune ROI**: Adjust `ROI_Y` / `ROI_X` (fractions of frame height/width)

## Gotchas
- Speed gate: a completed turn is discarded unless a speedometer reading (within `SPEED_LIMIT_KMH`) was recognised *during* the turn; the saved speed is the average of those readings.
- No tests / lint / typecheck configured
- YouTube URLs require internet + yt-dlp; local files need to exist
- AV1 codec streams will fail to decode in OpenCV (yt-dlp config excludes them); for local AV1 files run `pipeline.py --transcode` (system ffmpeg → cached H.264 in `<output>/.transcode_cache/`)
- Press `q` in preview window to stop early; `writer.wait_for_completion()` blocks until all clips finish encoding
