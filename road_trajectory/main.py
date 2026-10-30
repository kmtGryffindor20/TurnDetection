"""Road-trajectory mapper: split turn clip(s) in, 2D path out.

Usage:
    .\\venv\\Scripts\\python.exe road_trajectory\\main.py [clip-or-folder]

With no argument it maps VIDEO_PATH from trajectory_config.py (one clip from
dataset_turns_3class). Pass a folder to map every split turn clip inside it,
recursively. Each clip writes <Class>_<clip>.csv and <Class>_<clip>.png into
road_trajectory/output/ and (for a single clip) shows the plot at the end.

The speed fed to the odometry is read from the speedometer with the pipeline's
own EasyOCR recognizer (root speed_recognizer.py), falling back to a constant
only until the first valid reading arrives.
"""

import csv
import glob
import math
import os
import sys
from collections import deque

ROAD_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ROAD_DIR)
sys.path.insert(0, ROAD_DIR)
sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np

import config as pipeline_config  # root pipeline config, used by SpeedRecognizer
import trajectory_config as config
from speed_recognizer import SpeedRecognizer
from trajectory_mapper import RoadTrajectoryMapper
from trajectory_visualization import TrajectoryPreview, plot_trajectory

# EasyOCR model paths in the pipeline config are relative to the repo root;
# make them absolute so the tool works from any working directory.
if not os.path.isabs(pipeline_config.SPEED_MODEL_DIR):
    pipeline_config.SPEED_MODEL_DIR = os.path.join(
        ROOT_DIR, pipeline_config.SPEED_MODEL_DIR
    )


class SpeedSampler:
    """
    Throttled speedometer reads from the pipeline's EasyOCR recognizer.

    The readout is sampled every `SPEED_SAMPLE_STEP` frames and valid readings
    are averaged over the last `SPEED_AVERAGE_WINDOW` samples, so one misread
    frame cannot jerk the odometry. Until a reading arrives (or when speed
    recognition is disabled) it falls back to `SPEED_KMH`.
    """

    def __init__(self):
        self.reader = SpeedRecognizer() if config.USE_RECOGNISED_SPEED else None
        self.history = deque(maxlen=max(1, int(config.SPEED_AVERAGE_WINDOW)))
        self.bbox = None

    @property
    def speed_kmh(self):
        if self.history:
            return float(np.mean(self.history))
        return config.SPEED_KMH

    def update(self, frame, frame_index):
        """Sample if due -> (speed_kmh | None, readout bbox | None)."""
        if self.reader is None:
            return config.SPEED_KMH, None

        step = max(1, int(config.SPEED_SAMPLE_STEP))
        if frame_index % step == 0:
            # A miss clears the box, so a stale readout is never drawn.
            reading, self.bbox = self.reader.read(frame)
            valid = (
                reading is not None
                and config.SPEED_MIN_KMH < reading <= config.SPEED_MAX_KMH
            )
            if valid:
                self.history.append(reading)

        return self.speed_kmh, self.bbox


def map_trajectory(video_path, show_plot=None):
    """Runs the mapper over one split turn clip and saves its outputs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    mapper = RoadTrajectoryMapper(fps)
    preview = TrajectoryPreview()
    sampler = SpeedSampler()
    stem = _output_stem(video_path)

    print(
        f"Mapping {stem} ({fps:.1f} fps, {total} frames, "
        f"step {config.FRAME_STEP}, speed={'ocr' if sampler.reader else 'fixed'})..."
    )

    state = None
    speed_bbox = None
    frame_index = 0
    analysed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            speed_kmh, speed_bbox = sampler.update(frame, frame_index)

            if frame_index % config.FRAME_STEP == 0:
                state, _ = mapper.update(
                    frame, frame_index=frame_index, speed_kmh=speed_kmh
                )
                analysed += 1

            if not preview.show(frame, state, mapper.path, speed_bbox):
                print("Stopped by user.")
                break

            frame_index += 1
    finally:
        cap.release()
        preview.close()

    if not mapper.states:
        print(f"{stem}: no frames were analysed.")
        return mapper

    avg_speed = mapper.average_speed_kmh
    print(
        f"{stem}: {analysed} frames | distance {mapper.path_length:.0f} m | "
        f"net heading {math.degrees(mapper.bicycle.heading):+.1f} deg | "
        f"avg speed {avg_speed:.1f} km/h"
    )

    if config.SAVE_OUTPUT:
        _write_csv(mapper.states, stem)

    plot_path = (
        os.path.join(config.OUTPUT_DIR, f"{stem}.png")
        if config.SAVE_OUTPUT
        else None
    )
    plot_trajectory(
        mapper.path,
        plot_path,
        show=config.SHOW_PLOT if show_plot is None else show_plot,
        title=f"Trajectory - {stem}",
        distance=mapper.path_length,
        heading_deg=math.degrees(mapper.bicycle.heading),
        average_speed=avg_speed,
    )
    return mapper


def map_source(source):
    """Maps a single clip, or every clip inside a folder (recursively)."""
    if os.path.isdir(source):
        clips = sorted(
            glob.glob(os.path.join(source, "**", "*.mp4"), recursive=True)
        )
        if not clips:
            raise SystemExit(f"No .mp4 clips found in {source}")
        print(f"Mapping {len(clips)} split turn clips from {source}")
        mappers = []
        for i, clip in enumerate(clips, 1):
            print(f"\n[{i}/{len(clips)}] {os.path.relpath(clip, source)}")
            mappers.append(map_trajectory(clip, show_plot=False))
        print(f"\nMapped {len(mappers)} clips -> {config.OUTPUT_DIR}")
        return mappers

    return map_trajectory(source)


def _output_stem(video_path):
    """`Sharp/turn_0006.mp4` -> `Sharp_turn_0006`."""
    clip = os.path.splitext(os.path.basename(video_path))[0]
    folder = os.path.basename(os.path.dirname(os.path.abspath(video_path)))
    return f"{folder}_{clip}"


def _write_csv(states, stem):
    """Dumps the per-frame state so the trajectory can be re-plotted elsewhere."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(config.OUTPUT_DIR, f"{stem}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "time_s",
                "flow_x_px",
                "yaw_rad",
                "heading_rad",
                "x_m",
                "y_m",
                "speed_kmh",
            ]
        )
        for s in states:
            writer.writerow(
                [
                    s.frame_index,
                    f"{s.time_s:.3f}",
                    f"{s.flow_x:.4f}",
                    f"{s.yaw:.6f}",
                    f"{s.heading:.6f}",
                    f"{s.x:.4f}",
                    f"{s.y:.4f}",
                    f"{s.speed_kmh:.1f}",
                ]
            )

    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else config.VIDEO_PATH
    map_source(source)
