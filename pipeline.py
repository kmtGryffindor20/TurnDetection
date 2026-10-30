"""End-to-end batch runner: split turns for every video in a directory tree.

Walks a directory of group folders, points :mod:`config` at each video in
turn, and runs :func:`main.process_video_pipeline` sequentially. Progress is
shown with two tqdm bars: an outer ``Videos: n/N`` counter (with running
turns/clips totals) and an inner per-video frame bar (percentage/ETA) driven
by the pipeline's ``progress_callback``. ``--quiet`` silences the
per-turn/per-clip internal logs (same as ``config.VERBOSE = False``) while
keeping the bars and per-video result lines. Each video gets its own dataset
folder and Excel catalog so clip names and turn IDs never collide between
runs; a combined catalog for the whole batch is written at the end.

Usage
-----
    python pipeline.py <root_dir> [--output <dir>] [--no-preview]
                       [--skip-existing] [--dry-run] [--quiet] [--transcode]

Expected input layout (each immediate sub-directory is one group)
------------------------------------------------------------------
    <root_dir>/
        group_a/
            clip_01.mp4
            clip_02.mp4
        group_b/
            clip_01.mp4

Per-video output layout (``--output`` defaults to ./batch_output)
-----------------------------------------------------------------
    <output_dir>/
        group_a/clip_01/dataset_turns_3class/{Minor,Major,Sharp}/*.mp4
        group_a/clip_01/road_turn_dataset.xlsx
        group_b/clip_01/...
        road_turn_dataset.xlsx       <- combined catalog for the whole batch

Notes
-----
* ``config`` is mutated for the duration of each video and restored when the
  batch finishes, so the module-level defaults in ``config.py`` stay intact.
* One ``SpeedRecognizer`` is shared across the whole batch, so the EasyOCR
  models load once instead of once per video.
* ``--transcode`` handles sources this OpenCV build cannot decode (typically
  AV1, where OpenCV's bundled FFmpeg only exposes hardware decoding). Such
  files are converted once with system ``ffmpeg`` into
  ``<output>/.transcode_cache/`` as H.264 and reused on later runs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

import config
import main as turn_pipeline
from speed_recognizer import SpeedRecognizer

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpg", ".mpeg"}

CATALOG_NAME = "road_turn_dataset.xlsx"
DATASET_NAME = "dataset_turns_3class"
TRANSCODE_CACHE_NAME = ".transcode_cache"

# H.264 settings for sources OpenCV cannot decode (e.g. AV1 on older wheels).
TRANSCODE_CRF = 20
TRANSCODE_PRESET = "veryfast"

# Config values the batch runner overrides per video and restores afterwards.
OVERRIDDEN_CONFIG = (
    "INPUT_SOURCE",
    "DATASET_DIR",
    "EXCEL_FILE",
    "SHOW_PREVIEW",
    "VERBOSE",
)


@dataclass(frozen=True)
class VideoJob:
    """One video to process, together with where its outputs belong."""

    group: str
    path: Path
    output_dir: Path
    relative: Path  # source path relative to its group folder, suffix stripped
    decode_path: Path | None = None  # cached H.264 copy, when transcoded

    @property
    def label(self) -> str:
        return f"{self.group}/{self.path.name}"

    @property
    def input_path(self) -> Path:
        """File handed to OpenCV (the transcoded copy for AV1 sources)."""
        return self.decode_path if self.decode_path is not None else self.path

    @property
    def catalog_path(self) -> Path:
        return self.output_dir / CATALOG_NAME

    @property
    def done_marker(self) -> Path:
        return self.output_dir / ".done"


@dataclass
class JobResult:
    """Outcome of a single video run."""

    job: VideoJob
    status: str  # ok | no-turns | unreadable | missing | skipped | error
    records: pd.DataFrame | None = None
    error: str | None = None
    turns_detected: int = 0

    @property
    def clip_count(self) -> int:
        return 0 if self.records is None else len(self.records)


def discover_jobs(root_dir: Path, output_dir: Path) -> list[VideoJob]:
    """Finds every video under the immediate sub-directories of `root_dir`."""
    root_dir = root_dir.resolve()
    output_dir = output_dir.resolve()

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root_dir}")

    group_dirs = sorted(
        d for d in root_dir.iterdir() if d.is_dir() and d.resolve() != output_dir
    )
    if not group_dirs:
        raise ValueError(
            f"No group folders found in '{root_dir}'. Expected one sub-directory "
            "per group, each containing video files."
        )

    jobs: list[VideoJob] = []
    used_paths: set[str] = set()

    for group_dir in group_dirs:
        videos = sorted(
            p
            for p in group_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            tqdm.write(f"[skip] no videos found in '{group_dir}'")
            continue

        for video in videos:
            # Keep nested relative paths so same-named clips stay distinct.
            relative = video.relative_to(group_dir).with_suffix("")
            job_output = output_dir / group_dir.name / relative
            candidate, suffix = job_output, 2
            while str(candidate).lower() in used_paths:
                candidate = job_output.with_name(f"{job_output.name}_{suffix}")
                suffix += 1
            used_paths.add(str(candidate).lower())
            jobs.append(VideoJob(group_dir.name, video, candidate, relative))

    return jobs


def apply_config(job: VideoJob, show_preview: bool | None = None) -> None:
    """Points the shared `config` module at a single video's input/output."""
    config.INPUT_SOURCE = str(job.input_path)
    config.DATASET_DIR = str(job.output_dir / DATASET_NAME)
    config.EXCEL_FILE = str(job.catalog_path)
    if show_preview is not None:
        config.SHOW_PREVIEW = show_preview


def _probe_codec(path: Path) -> str | None:
    """Returns the first video stream's codec name via ffprobe, if available."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines[0].lower() if completed.returncode == 0 and lines else None


def _probe_duration(path: Path) -> float | None:
    """Returns the container duration in seconds via ffprobe, if available."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    try:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        duration = float(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return duration if duration > 0 else None


def _can_decode(path: Path) -> bool:
    """Cheap probe: can OpenCV open this file and decode its first frame?"""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        return bool(ok and frame is not None)
    finally:
        capture.release()


def _transcode_to_h264(
    job: VideoJob, cache_root: Path, codec: str | None
) -> tuple[VideoJob, str | None]:
    """Converts a source OpenCV cannot decode into a cached H.264 copy."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return job, (
            f"{job.label}: cannot decode {codec or 'this source'} and ffmpeg is "
            "not installed/on PATH (apt install ffmpeg)"
        )

    # Source suffix is kept in the cache name so clip.mp4 / clip.mkv (which map
    # to distinct output folders) cannot share one cached copy.
    cache_name = f"{job.relative.name}_{job.path.suffix.lstrip('.').lower()}.mp4"
    cache_path = cache_root / job.group / job.relative.parent / cache_name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        cache_path.is_file()
        and cache_path.stat().st_size > 0
        and cache_path.stat().st_mtime >= job.path.stat().st_mtime
    ):
        tqdm.write(f"[transcode] {job.label}: reusing cached H.264 copy")
        return replace(job, decode_path=cache_path), None

    tmp_path = cache_path.with_name(f"{cache_path.stem}.part{cache_path.suffix}")
    duration = _probe_duration(job.path)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(job.path),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-c:v", "libx264", "-preset", TRANSCODE_PRESET, "-crf", str(TRANSCODE_CRF),
        "-pix_fmt", "yuv420p", "-progress", "pipe:1", "-nostats",
        str(tmp_path),
    ]

    tqdm.write(f"[transcode] {job.label}: {codec or 'undecodable'} -> H.264 ...")
    bar = tqdm(
        total=duration,
        unit="s",
        desc=f"transcode {job.label}"[:70],
        position=1,
        leave=False,
        dynamic_ncols=True,
        file=sys.stderr,
    )
    process = None
    # stderr goes to a temp file: a chatty ffmpeg must not fill the pipe while
    # we are busy reading progress from stdout.
    with tempfile.TemporaryFile() as err_file:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=err_file,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in process.stdout or ():
                key, _, value = line.strip().partition("=")
                if key in {"out_time_us", "out_time_ms"} and value.isdigit():
                    # Both keys are microseconds in ffmpeg's -progress output.
                    seconds = int(value) / 1_000_000
                    done = min(seconds, duration) if duration else seconds
                    if done > bar.n:
                        bar.update(done - bar.n)
                elif key == "progress" and value == "end" and duration:
                    if duration > bar.n:
                        bar.update(duration - bar.n)
            returncode = process.wait()
            err_file.seek(0)
            stderr = err_file.read().decode("utf-8", "replace")
        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            tmp_path.unlink(missing_ok=True)
            raise
        except (OSError, subprocess.SubprocessError) as exc:
            if process is not None and process.poll() is None:
                process.kill()
            tmp_path.unlink(missing_ok=True)
            return job, f"transcode failed for {job.label}: {exc}"
        finally:
            bar.close()

    if returncode != 0 or not tmp_path.is_file():
        detail = [line for line in stderr.strip().splitlines() if line.strip()]
        message = detail[-1] if detail else f"ffmpeg exited with {returncode}"
        tmp_path.unlink(missing_ok=True)
        return job, f"transcode failed for {job.label}: {message}"

    tmp_path.replace(cache_path)
    tqdm.write(f"[transcode] {job.label}: using cached copy {cache_path}")
    return replace(job, decode_path=cache_path), None


def _prepare_source(
    job: VideoJob, cache_root: Path, transcode: bool
) -> tuple[VideoJob, str | None]:
    """Returns the job to run (with a decodable source) or an error message."""
    codec = _probe_codec(job.path)
    is_av1 = bool(codec and codec.startswith("av1"))

    if not transcode:
        if is_av1 and not _can_decode(job.path):
            tqdm.write(
                f"[warn] {job.label}: AV1 is not decodable by this OpenCV build; "
                "rerun with --transcode to cache an H.264 copy."
            )
        return job, None

    if not is_av1 and _can_decode(job.path):
        return job, None
    return _transcode_to_h264(job, cache_root, codec)


def run_video(
    job: VideoJob,
    show_preview: bool | None = None,
    speed_recognizer: SpeedRecognizer | None = None,
) -> JobResult:
    """Runs the existing single-video pipeline for one job."""
    if not job.path.is_file():
        return JobResult(job, "missing")

    frame_bar = None

    def on_progress(done: int, total: int) -> None:
        """Lazily creates and advances this video's frame-level progress bar."""
        nonlocal frame_bar
        if frame_bar is None:
            frame_bar = tqdm(
                total=total or None,
                desc=job.label[:60],
                unit="frame",
                position=1,
                leave=False,
                dynamic_ncols=True,
                file=sys.stderr,
            )
        delta = done - frame_bar.n
        if delta:
            frame_bar.update(delta)

    try:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        apply_config(job, show_preview)
        summary = (
            turn_pipeline.process_video_pipeline(
                str(job.input_path), speed_recognizer, on_progress
            )
            or {}
        )
        turns_detected = int(summary.get("turns_detected") or 0)

        if job.catalog_path.is_file():
            records = pd.read_excel(job.catalog_path)
            if not turns_detected:
                turns_detected = len(records)
            _mark_done(job, "ok", turns_detected, len(records))
            return JobResult(
                job,
                "ok" if len(records) else "no-turns",
                records=records,
                turns_detected=turns_detected,
            )

        # No catalog is written when no turn survived the speed/deflection gates.
        # A missing dataset folder as well means the stream could not even be read;
        # that is a failure, so no `.done` marker is written and a later run retries it.
        if Path(config.DATASET_DIR).is_dir():
            _mark_done(job, "no-turns", turns_detected, 0)
            return JobResult(job, "no-turns", turns_detected=turns_detected)
        return JobResult(job, "unreadable", turns_detected=turns_detected)
    except Exception:  # keep the batch going even if one video blows up
        return JobResult(job, "error", error=traceback.format_exc())
    finally:
        if frame_bar is not None:
            frame_bar.close()


def _mark_done(
    job: VideoJob, status: str, turns_detected: int = 0, clips_saved: int = 0
) -> None:
    """Records a completed run (with its turn counters) for `--skip-existing`."""
    payload = {
        "status": status,
        "turns_detected": turns_detected,
        "clips_saved": clips_saved,
    }
    try:
        job.done_marker.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    except OSError:
        pass


def _read_done_marker(job: VideoJob) -> dict:
    """Reads a job's `.done` marker, tolerating older plain-text markers."""
    try:
        data = json.loads(job.done_marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def combine_catalogs(results: list[JobResult], output_dir: Path) -> pd.DataFrame | None:
    """Merges the per-video catalogs into one batch-level Excel workbook."""
    frames = []
    for result in results:
        if result.records is None or result.records.empty:
            continue
        frame = result.records.copy()
        frame = frame.rename(columns={"Turn ID": "Turn ID (per video)"})
        frame.insert(0, "Video", result.job.path.name)
        frame.insert(0, "Source Video", str(result.job.path))
        frame.insert(0, "Group", result.job.group)
        frames.append(frame)

    if not frames:
        tqdm.write("No turns detected in the entire batch; no catalog written.")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined_path = output_dir / CATALOG_NAME
    combined.to_excel(combined_path, index=False)
    tqdm.write(f"Combined batch catalog saved to '{combined_path}'.")
    return combined


def report_result(result: JobResult) -> None:
    """Prints a one-line outcome for a single video."""
    job = result.job
    if result.status == "ok":
        breakdown = ""
        if "Class" in result.records.columns:
            classes = result.records["Class"].value_counts().to_dict()
            breakdown = " (" + ", ".join(
                f"{name}={count}" for name, count in classes.items()
            ) + ")"
        tqdm.write(
            f"[ok] {job.label}: {result.turns_detected} turn(s) detected, "
            f"{result.clip_count} clip(s){breakdown}"
        )
    elif result.status == "no-turns":
        if result.turns_detected:
            tqdm.write(
                f"[no clips] {job.label}: {result.turns_detected} turn(s) "
                "detected but none saved (speed gate / deflection discard)"
            )
        else:
            tqdm.write(f"[no turns] {job.label}")
    elif result.status == "unreadable":
        tqdm.write(f"[unreadable] {job.label}: stream could not be opened")
    elif result.status == "missing":
        tqdm.write(f"[missing] {job.label}: file not found")
    elif result.status == "skipped":
        suffix = (
            f" ({result.turns_detected} turn(s) detected)"
            if result.turns_detected
            else ""
        )
        tqdm.write(f"[skipped] {job.label}: already processed{suffix}")
    elif result.status == "error":
        tqdm.write(f"[error] {job.label}\n{result.error}")


def run_batch(
    root_dir: Path,
    output_dir: Path = Path("batch_output"),
    show_preview: bool | None = None,
    skip_existing: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
    transcode: bool = False,
) -> int:
    """Processes every video under `root_dir`, one after another."""
    root_resolved = root_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == root_resolved or root_resolved in output_resolved.parents:
        raise ValueError(
            "Output directory must live outside the input directory, otherwise "
            "extracted clips would be picked up as new videos on the next run."
        )

    jobs = discover_jobs(root_dir, output_dir)
    if not jobs:
        raise ValueError(f"No videos found under '{root_dir}'.")

    if dry_run:
        tqdm.write(f"Discovered {len(jobs)} video(s) under '{root_dir.resolve()}':")
        for job in jobs:
            tqdm.write(f"  {job.label} -> {job.output_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_config = {name: getattr(config, name) for name in OVERRIDDEN_CONFIG}
    if quiet:
        # Silence main/dataset_writer/stream_utils per-turn and per-clip logs;
        # the batch bar and per-video result lines below stay visible.
        config.VERBOSE = False
    speed_recognizer = SpeedRecognizer()
    results: list[JobResult] = []
    total_turns = 0
    total_clips = 0
    transcode_cache = output_dir / TRANSCODE_CACHE_NAME

    bar = tqdm(
        jobs,
        desc="Videos",
        unit="video",
        position=0,
        leave=True,
        dynamic_ncols=True,
        file=sys.stderr,
        # Video progress is just n/N; the active video has its own frame bar.
        bar_format="{desc}: {n_fmt}/{total_fmt}{postfix}",
    )
    try:
        for job in bar:
            marker = _read_done_marker(job)
            already_done = (
                job.done_marker.is_file() and marker.get("status") != "unreadable"
            )
            if skip_existing and (already_done or job.catalog_path.is_file()):
                # Keep previously produced clips in the combined catalog.
                records = None
                if job.catalog_path.is_file():
                    try:
                        records = pd.read_excel(job.catalog_path)
                    except Exception:
                        records = None
                turns_detected = marker.get("turns_detected")
                if not isinstance(turns_detected, int):
                    turns_detected = 0 if records is None else len(records)
                results.append(
                    JobResult(
                        job, "skipped", records=records, turns_detected=turns_detected
                    )
                )
                total_turns += turns_detected
                total_clips += 0 if records is None else len(records)
                bar.set_postfix(
                    turns=total_turns, clips=total_clips, refresh=False
                )
                report_result(results[-1])
                continue

            prepared, prep_error = _prepare_source(job, transcode_cache, transcode)
            if prep_error is not None:
                result = JobResult(job, "error", error=prep_error)
            else:
                result = run_video(prepared, show_preview, speed_recognizer)
            results.append(result)
            total_turns += result.turns_detected
            total_clips += result.clip_count
            bar.set_postfix(turns=total_turns, clips=total_clips, refresh=False)
            report_result(result)
    except KeyboardInterrupt:
        tqdm.write("\nBatch interrupted; keeping results processed so far.")
        bar.close()
        _restore_config(saved_config)
        results_left = len(jobs) - len(results)
        if results_left:
            tqdm.write(f"{results_left} video(s) were not processed.")
        combine_catalogs(results, output_dir)
        return 130
    finally:
        bar.close()
        _restore_config(saved_config)

    combined = combine_catalogs(results, output_dir)
    _print_summary(results, combined, output_dir)

    failed = {"error", "unreadable", "missing"}
    return 1 if any(r.status in failed for r in results) else 0


def _restore_config(saved: dict) -> None:
    for name, value in saved.items():
        setattr(config, name, value)


def _print_summary(
    results: list[JobResult], combined: pd.DataFrame | None, output_dir: Path
) -> None:
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    total_turns = sum(r.turns_detected for r in results)
    total_clips = sum(r.clip_count for r in results)
    tqdm.write("\nBatch complete.")
    tqdm.write(
        "Videos: "
        + ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
    )
    tqdm.write(f"Turns detected: {total_turns}")
    tqdm.write(f"Clips saved: {total_clips}")

    if combined is not None:
        class_counts = combined["Class"].value_counts().to_dict()
        breakdown = ", ".join(f"{name}={count}" for name, count in class_counts.items())
        tqdm.write(f"Class breakdown: {breakdown}")

    tqdm.write(f"Outputs: {output_dir.resolve()}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the turn-detection pipeline over every video in a directory "
            "of group folders, sequentially, with a progress bar."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example layout:\n"
            "  <root_dir>/group_a/clip_01.mp4\n"
            "  <root_dir>/group_b/clip_01.mp4\n\n"
            "Example:\n"
            "  python pipeline.py ./videos --output ./batch_output --no-preview\n"
        ),
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Directory whose immediate sub-directories each hold videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("batch_output"),
        help="Where per-video datasets and catalogs are written "
        "(default: ./batch_output).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the OpenCV preview window (headless/batch runs).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos already processed in an earlier run (per-video .done marker).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the discovered videos and their output folders, then exit.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Silence per-turn/per-clip internal logs (same as config.VERBOSE = False).",
    )
    parser.add_argument(
        "--transcode",
        "--transcode-av1",
        dest="transcode",
        action="store_true",
        help=(
            "Transcode AV1 (and any other source OpenCV cannot decode) into "
            "cached H.264 copies with ffmpeg, then process those (requires ffmpeg)."
        ),
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    show_preview = False if args.no_preview else None
    try:
        return run_batch(
            root_dir=args.root_dir,
            output_dir=args.output,
            show_preview=show_preview,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            quiet=args.quiet,
            transcode=args.transcode,
        )
    except (NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
