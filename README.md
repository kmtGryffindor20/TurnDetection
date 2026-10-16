# Turn Dataset Pipeline

Detects vehicle turn maneuvers from dashcam video (local file or YouTube URL)
via optical flow, and exports unannotated 20-second clips into a
3-class dataset (`Minor` / `Major` / `Sharp`), with an Excel catalog.

## Files

| File               | Responsibility                                              |
|--------------------|---------------------------------------------------------------|
| `config.py`        | All tunable constants (thresholds, durations, paths).         |
| `stream_utils.py`  | Resolves local/YouTube sources; `FrameReader` background thread. |
| `optical_flow.py`  | `TurnDetector` — Farneback or LK flow + turn state machine.   |
| `classification.py`| Deflection → `Minor`/`Major`/`Sharp` mapping.                  |
| `visualization.py` | HUD overlay drawing + `PreviewWindow` (main-thread only).      |
| `speed_ocr.py`     | `SpeedReader`/`TurnGate` — speedometer OCR gate (EasyOCR).     |
| `dataset_writer.py`| `DatasetWriter` — background clip encoding + Excel catalog.    |
| `main.py`          | Wires everything together; the pipeline entry point.           |

## Run

```bash
pip install -r requirements.txt
python main.py
```

Edit `INPUT_SOURCE` (and any thresholds) in `config.py`. Set
`SHOW_PREVIEW = False` for headless servers.

Farneback is used by default. Set `USE_LK_OPTICAL_FLOW = True` to use sparse
Lucas-Kanade tracking of feature points inside the configured ROI. LK tuning
options are also available in `config.py`.

## Speedometer OCR gate

A completed turn is only **saved** when numeric text is found in the lower part
of the frame. `TurnGate` OCRs the completion frame plus a few recent frames
(`OCR_GATE_SAMPLES`) and accepts the turn as soon as any of them yields digits;
the largest numeric token is written to the `Speed (km/h)` column of the Excel
catalog. A turn with no numeric text is reported as `rejected` by `TurnDetector`
and never saved. There is deliberately no plausibility (0–199) or confidence
check.

- Engine: **EasyOCR** — a learned text detector + recognizer. Its CRAFT
  detector returns **no boxes** on scenery (road/grass), so it can't invent a
  speed the way a plain digit-whitelisted OCR pass can.
- Region: `OCR_ROI_Y = (0.50, 0.88)`, `OCR_ROI_X = (0.15, 0.72)` — a band
  inside the lower half of the frame.
- `OCR_DETECT_MAG_RATIO` internally upscales the region so the small glyphs are
  detected. Only `0-9` is allowed (`OCR_DIGIT_ALLOWLIST`).
- The preview HUD draws a green box around the detected speed text (with a
  `SPEED <n>` label). It is kept live by a throttled pass
  (`OCR_LIVE_ENABLED`, `OCR_LIVE_INTERVAL_SECONDS`, default 0.5 s) that re-reads
  the frame while driving, independent of the turn gate.
- Set `OCR_ENABLED = False` to bypass the gate entirely.

### GPU / torch

`EASYOCR_GPU = True` needs a **CUDA build** of torch — a plain
`pip install easyocr` installs a CPU wheel. To switch:

```bash
pip install --force-reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

Model weights download on first run into `vendor/easyocr_models/`
(`config.EASYOCR_MODEL_DIR`, git-ignored). CPU only: set `EASYOCR_GPU = False`.

### OpenCV GUI (`namedWindow` not implemented)

EasyOCR depends on `opencv-python-headless`, which installs the same `cv2`
package **without** HighGUI — installing easyocr can silently break the preview
window (`error: ... The function is not implemented. Rebuild the library with
... GTK+ ... support`). Restore the GUI build (keep both packages installed but
put the GUI files back on top):

```bash
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps opencv-python
```


## Threading design

- **`FrameReader`** runs `cap.read()` on its own thread and pushes frames
  into a bounded queue. This isolates network/decode latency (especially
  for a resolved YouTube stream) from the analysis loop below — the
  analyzer is never blocked waiting on I/O. If the queue fills up (analysis
  falling behind), the oldest queued frame is dropped in favor of the
  newest, keeping the pipeline near-real-time rather than backing up
  indefinitely.
- **Main thread** pulls frames from that queue, runs `TurnDetector.update()`
  every `FRAME_STEP` frames, manages the rolling pre/post-turn buffer, and
  drives the OpenCV preview window. OpenCV's `imshow`/`waitKey` must stay on
  the main thread, which is why capture and encoding are pushed off of it
  instead.
- **`DatasetWriter`** uses a `ThreadPoolExecutor` (`CLIP_WRITER_WORKERS`,
  default 2) to encode finished 20-second clips to disk without blocking
  frame processing, and a lock-protected in-memory list for the catalog,
  flushed to Excel once at shutdown.

### Bugs fixed vs. the original single-file script

The original script captured post-turn frames with a **nested blocking
loop** (`cap.read()` called directly, bypassing the main loop). That left
`prev_gray` and the frame-step counter stale for ~10 seconds after every
saved clip, which could misfire the camera-cut filter and delay detection
of closely spaced turns. Here, every frame — pre-turn, mid-turn, or
post-turn — flows through the same reader queue and the same buffer;
`TurnDetector.sync_gray()` keeps the flow reference frame current during
post-roll capture without doing a full flow computation, so detection
state stays consistent as soon as post-roll capture ends.
