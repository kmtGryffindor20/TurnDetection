"""Background clip encoding and thread-safe dataset catalog logging."""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import pandas as pd

import config
from classification import classify_turn


class DatasetWriter:
    """
    Encodes 20-second clips to disk on a background thread pool so that
    VideoWriter I/O never blocks the frame-processing loop, and keeps a
    thread-safe in-memory catalog that is flushed to Excel at the end.
    """

    def __init__(self, fps, frame_size, max_workers: int = None):
        self.fps = fps
        self.frame_size = frame_size
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers or config.CLIP_WRITER_WORKERS,
            thread_name_prefix="clip-writer",
        )
        self._lock = threading.Lock()
        self._records = []
        self._turn_count = 0
        self._futures = []

        for class_name in config.CLASSES:
            os.makedirs(os.path.join(config.DATASET_DIR, class_name), exist_ok=True)

    def submit_clip(self, frames, accumulated_dx: float, average_speed: float = None):
        """
        Classifies the deflection and, if it clears the discard threshold,
        hands the frame sequence off to a background thread for encoding.
        `frames` should be the full pre+post frame list, in order.
        `average_speed` is the turn's mean speedometer reading (km/h).
        Returns the assigned turn ID, or None if the deflection was discarded.
        """
        turn_type = classify_turn(accumulated_dx)
        if turn_type is None:
            return None

        with self._lock:
            self._turn_count += 1
            turn_id = self._turn_count

        future = self.executor.submit(
            self._encode_clip, turn_id, turn_type, list(frames), accumulated_dx,
            average_speed,
        )
        self._futures.append(future)
        return turn_id

    def _encode_clip(self, turn_id, turn_type, frames, accumulated_dx, average_speed):
        clip_filename = f"turn_{turn_id:04d}.mp4"
        class_folder = os.path.join(config.DATASET_DIR, turn_type)
        clip_path = os.path.abspath(os.path.join(class_folder, clip_filename))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(clip_path, fourcc, self.fps, self.frame_size)
        try:
            for f in frames:
                out.write(f)
        finally:
            out.release()

        record = {
            "Turn ID": turn_id,
            "Class": turn_type,
            "Deflection Value": round(accumulated_dx, 2),
            "Average Speed (km/h)": (
                round(average_speed, 1) if average_speed is not None else None
            ),
            "Saved Clip Address": clip_path,
        }
        with self._lock:
            self._records.append(record)

        speed_text = f"{average_speed:.1f} km/h" if average_speed is not None else "n/a"
        if config.VERBOSE:
            print(
                f"Saved 20s Clip #{turn_id} ---> Class: '{turn_type}' "
                f"(Deflection: {accumulated_dx:.2f}, Avg Speed: {speed_text}) "
                f"| Path: {clip_path}"
            )
        return record

    @property
    def clip_count(self) -> int:
        """Number of clips encoded so far (accurate once writers have finished)."""
        with self._lock:
            return len(self._records)

    def wait_for_completion(self):
        """Blocks until every queued clip has finished encoding."""
        for future in self._futures:
            future.result()

    def flush_excel(self):
        with self._lock:
            records = list(self._records)
        if not records:
            if config.VERBOSE:
                print("\nProcessing complete! No distinct road turns were detected.")
            return
        records.sort(key=lambda r: r["Turn ID"])
        df = pd.DataFrame(records)
        df.to_excel(config.EXCEL_FILE, index=False)
        if config.VERBOSE:
            print(f"\nProcessing complete! Dataset catalog saved to '{config.EXCEL_FILE}'.")

    def shutdown(self):
        self.executor.shutdown(wait=True)
