# Road Trajectory Mapper

Turns an already-split turn clip into a 2D top-down path using dense optical
flow and a kinematic bicycle model, with the real speed read from the
speedometer. Independent of the turn-detection pipeline in the repo root — it
only shares the `venv` and the `SpeedRecognizer`.

## Quick start

```powershell
# default clip (Sharp/turn_0006.mp4 from dataset_turns_3class)
.\venv\Scripts\python.exe road_trajectory\main.py

# one clip
.\venv\Scripts\python.exe road_trajectory\main.py dataset_turns_3class\Minor\turn_0005.mp4

# every split turn clip in a folder (recursive)
.\venv\Scripts\python.exe road_trajectory\main.py dataset_turns_3class
```

The mapper is meant for the clips in `dataset_turns_3class/` (one turn, 20 s
each), not the raw driving videos. Pass a folder to map all of them; a single
clip opens the plot at the end, a folder writes all plots without windows.

Outputs:

```
road_trajectory/output/
  Sharp_turn_0006.png   # vertical path, forward = up
  Sharp_turn_0006.csv   # per-frame frame/time/flow_x/yaw/heading/x/y/speed
```

Press `q` in the preview window to stop early. Set `SHOW_PREVIEW = False` for
a headless run.

## How it works

| Step | Where |
|------|-------|
| Resize to 640x360, grayscale, blank above the horizon | `FlowYawEstimator._prepare` |
| Farneback dense flow, mean horizontal displacement over the bottom half | `FlowYawEstimator.update` |
| Rolling average (`SMOOTHING_WINDOW`) + deadband (`DEADBAND_PX`) | `FlowYawEstimator.update` |
| Pixels -> radians (`PIXELS_TO_RADIANS`) with `YAW_SIGN` | `FlowYawEstimator.update` |
| Speedometer OCR every `SPEED_SAMPLE_STEP` frames, averaged | `SpeedSampler` in `main.py` |
| Heading + position integration | `KinematicBicycle.step` |
| Orchestration / path history | `RoadTrajectoryMapper.update` |
| Plot, live preview, path inset | `trajectory_visualization.py` |

### Orientation and sign convention

Everything is drawn top-down with the bike travelling **up the page**:

```
+Y = forward (up)        +X = right (lateral)
heading 0 = straight ahead, positive heading = LEFT turn
x -= v * sin(heading)    y += v * cos(heading)
```

The camera is the moving observer, so the road streams the opposite way to the
turn. `YAW_SIGN = -1` is what puts left turns on the left of the plot; flip it
to `+1` if a known turn comes out mirrored.

The matplotlib plot keeps `set_aspect("equal")`, so a 90-degree turn can never
render as a shallow curve, and adds a minimum plotting width so an almost
straight clip does not collapse into a hairline.

## Tuning

- **`PIXELS_TO_RADIANS`** (default `0.001`) — the calibration constant. If a
  known 90-degree turn comes out as ~45 degrees, double it. It depends on
  camera FOV, resolution and how much of the frame is road, so it is the single
  most important knob.
- **`YAW_SIGN`** (default `-1`) — left/right mirror.
- **`DEADBAND_PX`** (default `0.5`) — raise it (1.0–2.0) if the path wanders
  while riding straight; lower it if gentle bends disappear.
- **`SMOOTHING_WINDOW`** (default `5`) — longer window = smoother but laggier.
- **`ROI_Y` / `ROI_X`** — where the road is. On this motorcycle POV footage the
  bottom of the frame is cockpit, mirrors and gloves, so the plan's
  full-width bottom half is used; narrow it (e.g. `ROI_X = (0.20, 0.80)`) to
  exclude the mirrors if passing traffic starts dragging the path sideways.
- **`USE_RECOGNISED_SPEED`** (default `True`) — `False` uses `SPEED_KMH` for
  every frame instead of the OCR reading.
- **`SPEED_SAMPLE_STEP`** (default `10`) — OCR cadence. Lower = more responsive
  speed, higher = faster.
- **`SPEED_KMH`** (default `40`) — fallback only, used until the first valid
  speedometer reading arrives (and whenever recognition is disabled).
- **`FRAME_STEP`** (default `1`) — analyse every Nth frame. `2`–`3` roughly
  halves/thirds the runtime; the bicycle model accounts for the time gap.

Reference throughput on this machine: ~70 s per 20 s clip at `FRAME_STEP = 1`
(~36 ms flow + ~4 ms amortised OCR per frame). Mapping the whole
`dataset_turns_3class/` (82 clips) takes roughly 1.5 hours.

## Files

```
trajectory_config.py        # all tunables
trajectory_mapper.py        # FlowYawEstimator, KinematicBicycle, RoadTrajectoryMapper
trajectory_visualization.py # matplotlib plot, live preview, path inset
main.py                     # entry point: clip(s) -> CSV + PNG, SpeedSampler
flow.py                     # original scratch script (kept for reference)
```
