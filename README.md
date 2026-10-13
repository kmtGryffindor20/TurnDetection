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
