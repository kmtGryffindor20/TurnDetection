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

Speedometer OCR gate
    A completed turn is only saved when a speedometer reading is found in the
    lower part of the frame (see speed_ocr.py). The reading is stored as the
    clip's speed in the Excel catalog.

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

import config
from stream_utils import resolve_video_source, FrameReader
from optical_flow import TurnDetector
from dataset_writer import DatasetWriter
from visualization import PreviewWindow
from speed_ocr import SpeedReader, TurnGate


def process_video_pipeline(source_path: str):
    stream_url = resolve_video_source(source_path)

    reader = FrameReader(stream_url).start()
    fps, width, height = reader.fps, reader.width, reader.height

    pre_buffer_size = max(fps * config.PRE_TURN_SECONDS, 1)
    post_buffer_size = max(fps * config.POST_TURN_SECONDS, 1)
    pre_frame_buffer = deque(maxlen=pre_buffer_size)

    first_frame = reader.read()
    if first_frame is None:
        print("Error: Video stream is empty or unreadable.")
        reader.stop()
        return

    pre_frame_buffer.append(first_frame.copy())

    # OCR-based turn gate: a completed turn is only saved when numeric text is
    # found in the lower part of the frame. The readout is only legible on a
    # fraction of frames, so the gate also OCRs a few recent pre-turn frames.
    def recent_frames(count: int):
        """Up to `count` recent pre-turn frames, newest first."""
        buf = list(pre_frame_buffer)
        if not buf:
            return []
        step = max(1, len(buf) // (count + 1))
        picks = []
        i = len(buf) - 1 - step
        while len(picks) < count and i >= 0:
            picks.append(buf[i])
            i -= step
        return picks

    gate = None
    speed_reader = None
    if config.OCR_ENABLED:
        speed_reader = SpeedReader()
        if not speed_reader.available:
            print(
                f"[OCR] Warning: OCR unavailable -> no turns will be saved "
                f"({speed_reader.unavailable_reason})"
            )
        gate = TurnGate(speed_reader, frame_provider=recent_frames)

    # Throttled live OCR interval (frames) for the HUD; 0 disables it.
    live_interval = (
        max(1, int(fps * config.OCR_LIVE_INTERVAL_SECONDS))
        if (speed_reader is not None and speed_reader.available and config.OCR_LIVE_ENABLED)
        else 0
    )

    detector = TurnDetector(first_frame, completion_validator=gate)
    writer = DatasetWriter(fps=fps, frame_size=(width, height))
    preview = PreviewWindow()

    capturing_post_roll = False
    post_roll_frames = []
    post_roll_accum_dx = 0.0
    post_roll_speed = None
    pending_pre_snapshot = []
    ocr_bbox = None
    ocr_text = None

    frame_idx = 0
    print("Analyzing video stream... Press 'q' inside the preview window to stop early.")

    try:
        while True:
            frame = reader.read()
            if frame is None:
                break
            frame_idx += 1
            pre_frame_buffer.append(frame.copy())

            if capturing_post_roll:
                post_roll_frames.append(frame.copy())
                detector.sync_gray(frame)
                if len(post_roll_frames) >= post_buffer_size:
                    full_clip = pending_pre_snapshot + post_roll_frames
                    writer.submit_clip(full_clip, post_roll_accum_dx, post_roll_speed)
                    capturing_post_roll = False
                    post_roll_frames = []
                    pending_pre_snapshot = []
                    post_roll_speed = None
            elif frame_idx % config.FRAME_STEP == 0:
                event, deflection = detector.update(frame)
                if event == "completed":
                    pending_pre_snapshot = list(pre_frame_buffer)
                    post_roll_accum_dx = deflection
                    post_roll_speed = gate.speed if gate else None
                    post_roll_frames = []
                    capturing_post_roll = True
                elif event == "rejected":
                    print(
                        "Turn rejected: no numeric speed text found in the "
                        "lower part of the frame."
                    )
                if gate is not None and not live_interval:
                    # Fallback HUD box when the live OCR pass is disabled.
                    ocr_bbox = gate.last.bbox
                    ocr_text = gate.last.text

            # Live OCR for the HUD (throttled): keeps the speed box and reading
            # tracking the readout instead of updating only at turn completion.
            if live_interval and frame_idx % live_interval == 0:
                live = speed_reader.read(frame)
                ocr_bbox = live.bbox
                ocr_text = live.text if live.detected else None

            is_turning_now = detector.is_turning or capturing_post_roll
            if not preview.show(
                frame, detector.current_dx, detector.current_dy, is_turning_now,
                ocr_bbox, ocr_text,
            ):
                print("Execution manually stopped by user.")
                break
    finally:
        reader.stop()
        preview.close()
        writer.wait_for_completion()
        writer.flush_excel()
        writer.shutdown()


if __name__ == "__main__":
    process_video_pipeline(config.INPUT_SOURCE)
