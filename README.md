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
| `speed_recognizer.py` | `SpeedRecognizer` — EasyOCR speedometer readout (speed gate). |
| `visualization.py` | HUD overlay drawing + `PreviewWindow` (main-thread only).      |
| `dataset_writer.py`| `DatasetWriter` — background clip encoding + Excel catalog.    |
| `main.py`          | Wires everything together; single-video entry point.           |
| `pipeline.py`      | Batch runner: every video in a directory tree, sequentially.   |

## Run

```bash
pip install -r requirements.txt
python main.py
```

Edit `INPUT_SOURCE` (and any thresholds) in `config.py`. Set
`SHOW_PREVIEW = False` for headless servers.

## Server / VM install

Targets Python 3.10–3.12 (developed on 3.11). On a headless Linux VM:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv ffmpeg   # ffmpeg optional, for yt-dlp

git clone <repo> && cd TurnDetection
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# NVIDIA GPU build (check your driver with `nvidia-smi` first):
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU-only alternative:
# python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements-server.txt
```

`requirements-server.txt` uses `opencv-python-headless`, so no GUI/`libGL`
packages are needed — always run with `--no-preview` (or set
`SHOW_PREVIEW = False`). Sanity check the install:

```bash
python -c "import cv2, torch, easyocr, pandas; print(cv2.__version__, torch.__version__, torch.cuda.is_available())"
```

EasyOCR downloads its models into `vendor/easyocr/` on first use (~95 MB). On
an offline VM, copy `vendor/easyocr/` over from a machine that already has it.
Run commands from the repo root so the relative `SPEED_MODEL_DIR` resolves.

Set `SPEED_GPU = True` only when `torch.cuda.is_available()` is `True`,
otherwise `SPEED_GPU = False`. For long batches use `tmux`, `nohup`, or a
systemd unit; per-video outputs plus `--skip-existing` make a disconnect safe
to resume:

```bash
nohup python pipeline.py ./videos --output ./batch_output --no-preview \
    --skip-existing > batch.log 2>&1 &
```

## Batch processing

Process a directory of group folders (each holding one or more videos) in one
go. `pipeline.py` points `config` at each video, runs `main.py`'s pipeline
sequentially, and shows two tqdm bars: `Videos: n/N` for the batch (plus
running turns-detected/clips-saved totals) and a per-video frame bar
(percentage/ETA) for the video being processed:

```bash
python pipeline.py ./videos --output ./batch_output --no-preview
```

Expected input layout and per-video outputs:

```
videos/                          batch_output/
  group_a/                         group_a/
    clip_01.mp4                      clip_01/
    clip_02.mp4                        dataset_turns_3class/{Minor,Major,Sharp}/
  group_b/                             road_turn_dataset.xlsx
    clip_01.mp4                      group_b/clip_01/...
                                   road_turn_dataset.xlsx   # combined catalog
```

Each video gets its own dataset folder and catalog, so turn IDs and clip names
never collide. The combined catalog adds `Group`, `Source Video`, and
`Video` columns. The batch summary reports **turns detected** (completed flow
manoeuvres, before the speed/deflection gates) separately from **clips saved**.

Useful flags: `--skip-existing` (resume a batch), `--dry-run` (list discovered
videos and output folders), `--no-preview` (headless), and `-q/--quiet`
(silence per-turn/per-clip internal logs such as "Turn rejected: no speed…"
and "Saved 20s Clip #…"; the bars and per-video result lines stay). Quiet mode
can also be set globally with `VERBOSE = False` in `config.py`. The process
exits non-zero if any video fails.

Farneback is used by default. Set `USE_LK_OPTICAL_FLOW = True` to use sparse
Lucas-Kanade tracking of feature points inside the configured ROI. LK tuning
options are also available in `config.py`.

## Speed gate

A completed optical-flow turn is only saved when a speedometer reading is
recognised *during* the turn (see `speed_recognizer.py`). The readout is
located with EasyOCR inside `SPEED_ROI_Y`/`SPEED_ROI_X` (lower-middle console),
recognised every `SPEED_SAMPLE_STEP` frames, and any reading above
`SPEED_LIMIT_KMH` is rejected as noise. Readings collected while the turn is
active are averaged, and that average becomes the clip's `Average Speed (km/h)`
in the Excel catalog. Turns with no valid reading are discarded.

Set `SPEED_ENABLED = False` to disable the gate and save every turn (the speed
column is then blank). GPU recognition needs a CUDA build of `torch`; set
`SPEED_GPU = False` to run on CPU.

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
