# TurnDetection — Agent Quick Reference

## Project Overview
Pipeline that detects vehicle turns from dashcam video (local file or YouTube URL) using dense optical flow (Farneback), exports 20-second clips (10s pre + 10s post) into a 3-class dataset (`Minor`/`Major`/`Sharp`), and writes an Excel catalog.

## Entry Point
```
python main.py
```
Edit `INPUT_SOURCE` in `config.py` (local path or YouTube URL). Set `SHOW_PREVIEW = False` for headless servers.

## Dependencies
```
pip install -r requirements.txt   # opencv-python, yt-dlp, pandas, numpy, easyocr
```
Speedometer OCR uses **EasyOCR** (learned text detector + recognizer), which is
what stops scenery from being misread as a speed: its detector returns no boxes
when there is no text. `EASYOCR_GPU = True` needs a **CUDA build** of torch (the
default `pip install easyocr` gives a CPU wheel); model weights download on
first run into `vendor/easyocr_models/` (git-ignored).

## Architecture (Threading Model)
| Component | Thread | Responsibility |
|-----------|--------|----------------|
| `FrameReader` | Background (daemon) | Reads/decodes frames into bounded queue; drops oldest frame when full to stay near-real-time |
| Main thread | Main | Pulls frames, runs `TurnDetector.update()`, manages rolling buffer, drives OpenCV preview (`imshow`/`waitKey` must stay on main thread) |
| `SpeedReader`/`TurnGate` | Main | On turn completion, EasyOCR-reads the completion frame + recent buffered frames; gates the turn and yields the speed |
| `DatasetWriter` | ThreadPoolExecutor (`CLIP_WRITER_WORKERS=2`) | Encodes clips to disk in background; maintains thread-safe catalog flushed to Excel on shutdown |

## Key Files
- `config.py` — All tunable constants (thresholds, durations, paths, ROI, threading)
- `stream_utils.py` — `resolve_video_source()` (YouTube → direct URL via yt-dlp, excludes AV1), `FrameReader` class
- `optical_flow.py` — `TurnDetector` state machine; optional `completion_validator(frame)` hook rejects a completed turn; call `sync_gray(frame)` during post-roll capture to keep `prev_gray` current without full flow computation
- `classification.py` — `classify_turn(deflection)` → `None`/`Minor`/`Major`/`Sharp`
- `speed_ocr.py` — `SpeedReader` (EasyOCR on GPU) + `TurnGate` (samples recent frames; returns `True` when numeric text is found)
- `dataset_writer.py` — `DatasetWriter` background encoding + Excel catalog
- `visualization.py` — HUD overlay + `PreviewWindow` (main-thread only); draws a green box around the last detected speed text
- `main.py` — Wires everything together

## Critical Design Notes
1. **Post-turn capture uses `sync_gray()`** — While capturing post-turn frames, `TurnDetector.sync_gray(frame)` advances the reference frame without running optical flow. This prevents the next `update()` from diffing frames seconds apart (which would trigger false camera-cut resets).
2. **Frame drop policy** — `FrameReader` drops oldest queued frame when full. This keeps pipeline current; it does NOT guarantee every frame is processed.
3. **OpenCV HighGUI** — Preview window (`cv2.imshow`/`waitKey`) MUST run on main thread. Never call from worker threads.
4. **YouTube codec selection** — yt-dlp format string prefers H.264 (avc1) / VP9, explicitly excludes AV1 (av01) which OpenCV often cannot decode.
5. **OCR turn gate** — `TurnDetector` accepts an optional `completion_validator(frame) -> bool`. `main.py` passes a `TurnGate` that EasyOCR-reads the completion frame plus up to `OCR_GATE_SAMPLES-1` recent pre-turn frames; a turn is accepted as soon as one yields numeric text (the largest numeric token is used as the speed), otherwise it is reported as `"rejected"` and never saved. There is no plausibility/confidence check — EasyOCR's learned detector is what prevents scenery (road/grass) from being read as a speed, and at 640×360 the readout is legible on only a fraction of frames, hence the multi-frame sample. A separate throttled pass (`OCR_LIVE_INTERVAL_SECONDS`) re-reads the current frame while driving and drives the HUD speed box (see `visualization.py`).

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
| `OCR_ENABLED` | True | Master switch for the OCR turn gate |
| `OCR_ROI_Y` / `OCR_ROI_X` | (0.50, 0.88) / (0.15, 0.72) | Speedometer OCR band inside the lower half of the frame |
| `OCR_DETECT_MAG_RATIO` | 3.0 | Internal upscale so the small glyphs are detected |
| `OCR_GATE_SAMPLES` | 5 | Frames (completion + recent) OCR'd per turn; accepts if any yields digits |
| `OCR_LIVE_ENABLED` / `OCR_LIVE_INTERVAL_SECONDS` | True / 0.5 | Throttled live OCR that keeps the HUD speed box up to date |
| `EASYOCR_GPU` / `EASYOCR_MODEL_DIR` | True / vendor/easyocr_models | GPU needs a CUDA torch build; models stored locally |


## Output Structure
```
dataset_turns_3class/
  Minor/
  Major/
  Sharp/
road_turn_dataset.xlsx  # Catalog with Turn ID, Class, Speed (km/h), Deflection Value, Saved Clip Address
```

## Common Tasks
- **Change input source**: Edit `config.INPUT_SOURCE`
- **Run headless**: Set `config.SHOW_PREVIEW = False`
- **Adjust sensitivity**: Modify `TURN_THRESHOLD`, `MIN_TURN_DURATION`, classification boundaries
- **Change clip length**: Modify `PRE_TURN_SECONDS` / `POST_TURN_SECONDS`
- **Tune ROI**: Adjust `ROI_Y` / `ROI_X` (fractions of frame height/width)
- **Tune OCR / speedometer**: Adjust `OCR_ROI_Y` / `OCR_ROI_X`, `OCR_DETECT_MAG_RATIO`, `OCR_GATE_SAMPLES`; disable the gate with `OCR_ENABLED = False`; GPU on/off via `EASYOCR_GPU`

## Gotchas
- `requirements.txt` covers the Python deps, but the **CUDA torch build must be installed separately** for GPU EasyOCR (see Dependencies)
- EasyOCR's `opencv-python-headless` dependency can override the GUI `cv2` and break the preview (`namedWindow ... not implemented`); fix with `pip uninstall -y opencv-python-headless` then `pip install --force-reinstall --no-deps opencv-python`
- No tests / lint / typecheck configured
- YouTube URLs require internet + yt-dlp; local files need to exist
- AV1 codec streams will fail to decode in OpenCV (yt-dlp config excludes them)
- Press `q` in preview window to stop early; `writer.wait_for_completion()` blocks until all clips finish encoding
