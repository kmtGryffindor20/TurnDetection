"""Background clip encoding and thread-safe dataset catalog logging."""
import functools
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import pandas as pd

import config
from classification import classify_turn

_FFMPEG_WARNING_SHOWN = False


@functools.lru_cache(maxsize=1)
def _find_ffmpeg():
    """Return an ffmpeg path that supports libx264, or None."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "-hide_banner", "-loglevel", "error", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return exe if "libx264" in result.stdout else None


class DatasetWriter:
    """
    Encodes 20-second clips to disk on a background thread pool so that
    encoding never blocks the frame-processing loop, and keeps a
    thread-safe in-memory catalog that is flushed to Excel at the end.
    """

    def __init__(self, fps, frame_size, max_workers: int = None):
        global _FFMPEG_WARNING_SHOWN

        self.fps = fps
        self.frame_size = frame_size
        self._ffmpeg = _find_ffmpeg()
        if self._ffmpeg is None and not _FFMPEG_WARNING_SHOWN:
            _FFMPEG_WARNING_SHOWN = True
            if config.VERBOSE:
                print(
                    "[dataset] ffmpeg with libx264 not found; falling back to "
                    "OpenCV mp4v (clips will be much larger). Install ffmpeg "
                    "to reduce disk usage."
                )
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

        if self._ffmpeg:
            try:
                self._encode_h264(clip_path, frames)
            except Exception as exc:  # noqa: BLE001 - keep a working fallback
                if config.VERBOSE:
                    print(
                        f"[dataset] H.264 encode failed for {clip_filename} "
                        f"({exc}); falling back to mp4v."
                    )
                self._encode_mp4v(clip_path, frames)
        else:
            self._encode_mp4v(clip_path, frames)

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

    def _encode_h264(self, clip_path, frames):
        """Pipe BGR frames to ffmpeg/libx264 (much smaller than mp4v)."""
        width, height = self.frame_size
        cmd = [
            self._ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", config.CLIP_PRESET,
            "-crf", str(config.CLIP_CRF),
            "-pix_fmt", "yuv420p",
            "-threads", str(config.CLIP_FFMPEG_THREADS),
            "-movflags", "+faststart",
            clip_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        stderr_chunks = []

        def drain_stderr():
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)

        drainer = threading.Thread(target=drain_stderr, daemon=True)
        drainer.start()

        error = None
        try:
            for frame in frames:
                if frame.shape[1] != width or frame.shape[0] != height:
                    raise ValueError(
                        f"frame is {frame.shape[1]}x{frame.shape[0]}, "
                        f"expected {width}x{height}"
                    )
                proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001 - reported below
            error = exc
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass

        returncode = proc.wait()
        drainer.join(timeout=5)
        stderr = b"".join(stderr_chunks).decode("utf-8", "replace").strip()

        if error is not None:
            self._remove_partial(clip_path)
            raise error
        if returncode != 0:
            self._remove_partial(clip_path)
            raise RuntimeError(stderr or f"ffmpeg exited with code {returncode}")

    def _encode_mp4v(self, clip_path, frames):
        """Fallback encoder used when ffmpeg/libx264 is unavailable."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(clip_path, fourcc, self.fps, self.frame_size)
        if not out.isOpened():
            raise RuntimeError(f"OpenCV could not open VideoWriter for {clip_path}")
        try:
            for frame in frames:
                out.write(frame)
        finally:
            out.release()

    @staticmethod
    def _remove_partial(path):
        try:
            os.remove(path)
        except OSError:
            pass

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
