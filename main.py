"""
Entry point: wires together threaded frame capture, optical-flow turn
detection, live HUD preview, and background dataset-clip encoding.

Threading model
----------------
FrameReader (background thread)
    Reads/decodes frames from cv2.VideoCapture into a bounded queue.
    Isolates slow network/decode I/O -- especially relevant for a resolved
    YouTube stream -- from the analysis loop.

main thread
    Pulls frames from the queue, runs optical flow via TurnDetector,
    manages the rolling pre/post-turn frame buffer, and drives the OpenCV
    preview window (OpenCV's HighGUI calls must stay on the main thread).

DatasetWriter (thread pool)
    Encodes finished 20-second clips to disk and updates the in-memory
    catalog concurrently, so VideoWriter I/O never stalls frame processing.

Speed gate
    A completed turn is only saved when a speedometer reading within
    SPEED_LIMIT_KMH was recognised during the turn (see speed_recognizer.py).
    The reading is sampled while the turn is active, averaged over the turn's
    duration, and stored as the clip's speed in the Excel catalog.

Buffering note
----------------
Every frame -- whether a turn is active or not -- flows through the same
queue and the same rolling `pre_frame_buffer`. While a clip's post-turn
frames are being collected, TurnDetector.sync_gray() keeps its internal
reference frame current without running a full flow computation, so
analysis can resume immediately once the clip is captured (this avoids
comparing frames that are seconds apart, which is what previously
triggered spurious camera-cut resets).
"""
from collections import deque
import time

import config
from stream_utils import resolve_video_source, FrameReader
from optical_flow import TurnDetector
from dataset_writer import DatasetWriter
from visualization import PreviewWindow
from speed_recognizer import SpeedRecognizer


def process_video_pipeline(source_path: str, speed_recognizer=None, progress_callback=None):
    """
    Processes one video end to end.

    `speed_recognizer` lets a caller reuse a single EasyOCR-backed recognizer
    across videos (batch mode). `progress_callback`, when given, is called as
    `progress_callback(done_frames, total_frames)` after every frame; total is
    0 when the container does not report a frame count.
    """
    stream_url = resolve_video_source(source_path)

    reader = FrameReader(stream_url).start()
    fps, width, height = reader.fps, reader.width, reader.height
    if progress_callback is not None:
        progress_callback(0, reader.frame_count)

    pre_buffer_size = max(fps * config.PRE_TURN_SECONDS, 1)
    post_buffer_size = max(fps * config.POST_TURN_SECONDS, 1)
    pre_frame_buffer = deque(maxlen=pre_buffer_size)

    first_frame = reader.read()
    if first_frame is None:
        print("Error: Video stream is empty or unreadable.")
        reader.stop()
        return

    pre_frame_buffer.append(first_frame.copy())
    detector = TurnDetector(first_frame)
    # A shared recognizer can be passed in (batch mode) so the EasyOCR models
    # are loaded once instead of per video.
    speed_recognizer = speed_recognizer or SpeedRecognizer()
    if config.SPEED_ENABLED:
        # Load the models before the timed loop so the one-off startup cost is
        # reported separately and does not inflate the OCR stage timings.
        _ = speed_recognizer.reader
    writer = DatasetWriter(fps=fps, frame_size=(width, height))
    preview = PreviewWindow()

    capturing_post_roll = False
    post_roll_frames = []
    post_roll_accum_dx = 0.0
    post_roll_average_speed = None
    pending_pre_snapshot = []

    # Speed readings collected across the duration of the current turn.
    turn_speeds = []
    hud_speed = None
    hud_bbox = None

    frame_idx = 0
    # A turn "detected" is any completed flow manoeuvre; it is only saved as a
    # clip if it clears the speed gate and the classification discard threshold.
    turns_detected = 0
    stage_seconds = {"read": 0.0, "flow": 0.0, "ocr": 0.0}
    stage_calls = {"read": 0, "flow": 0, "ocr": 0}
    loop_started = time.perf_counter()
    if config.VERBOSE:
        print("Analyzing video stream... Press 'q' inside the preview window to stop early.")

    try:
        while True:
            t0 = time.perf_counter()
            frame = reader.read()
            stage_seconds["read"] += time.perf_counter() - t0
            stage_calls["read"] += 1
            if frame is None:
                break
            frame_idx += 1
            if progress_callback is not None:
                progress_callback(frame_idx, reader.frame_count)
            pre_frame_buffer.append(frame.copy())

            if capturing_post_roll:
                post_roll_frames.append(frame.copy())
                detector.sync_gray(frame)
                if len(post_roll_frames) >= post_buffer_size:
                    full_clip = pending_pre_snapshot + post_roll_frames
                    writer.submit_clip(
                        full_clip, post_roll_accum_dx, post_roll_average_speed
                    )
                    capturing_post_roll = False
                    post_roll_frames = []
                    pending_pre_snapshot = []
                    post_roll_average_speed = None
            elif frame_idx % config.FRAME_STEP == 0:
                t0 = time.perf_counter()
                event, deflection = detector.update(frame)
                stage_seconds["flow"] += time.perf_counter() - t0
                stage_calls["flow"] += 1
                if event == "ongoing" and detector.turn_step_counter == 1:
                    turn_speeds = []  # a new turn just started
                elif event == "completed":
                    turns_detected += 1
                    average_speed = (
                        sum(turn_speeds) / len(turn_speeds) if turn_speeds else None
                    )
                    if average_speed is None:
                        if config.VERBOSE:
                            print(
                                "Turn rejected: no speed within the limit was "
                                "detected during the turn."
                            )
                    else:
                        pending_pre_snapshot = list(pre_frame_buffer)
                        post_roll_accum_dx = deflection
                        post_roll_average_speed = average_speed
                        post_roll_frames = []
                        capturing_post_roll = True
                    turn_speeds = []

            # Speed recognition: throttled, feeds the live HUD everywhere and is
            # accumulated across the turn's duration for the saved average.
            if frame_idx % config.SPEED_SAMPLE_STEP == 0:
                t0 = time.perf_counter()
                speed, bbox = speed_recognizer.read(frame)
                stage_seconds["ocr"] += time.perf_counter() - t0
                stage_calls["ocr"] += 1
                hud_speed, hud_bbox = speed, bbox
                if (
                    speed is not None
                    and detector.is_turning
                    and 0 < speed <= config.SPEED_LIMIT_KMH
                ):
                    turn_speeds.append(speed)

            is_turning_now = detector.is_turning or capturing_post_roll
            if not preview.show(
                frame, detector.current_dx, detector.current_dy, is_turning_now,
                hud_speed, hud_bbox,
            ):
                if config.VERBOSE:
                    print("Execution manually stopped by user.")
                break
    finally:
        loop_elapsed = time.perf_counter() - loop_started
        reader.stop()
        preview.close()
        writer.wait_for_completion()
        writer.flush_excel()
        writer.shutdown()

    if config.VERBOSE:
        accounted = sum(stage_seconds.values())
        parts = [
            f"{name} {stage_seconds[name]:.1f}s/{stage_calls[name]}"
            f" ({stage_seconds[name] / stage_calls[name] * 1000:.0f} ms)"
            for name in ("read", "flow", "ocr")
            if stage_calls[name]
        ]
        parts.append(f"other {loop_elapsed - accounted:.1f}s")
        print(f"Stage timing ({loop_elapsed:.1f}s): " + ", ".join(parts))

    # Consumed by the batch runner for its running/summary counters.
    return {
        "frames": frame_idx,
        "turns_detected": turns_detected,
        "clips_saved": writer.clip_count,
    }


if __name__ == "__main__":
    process_video_pipeline(config.INPUT_SOURCE)
