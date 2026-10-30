"""Video source resolution and threaded frame capture."""
import os
import queue
import threading

import cv2

import config


def resolve_video_source(source: str) -> str:
    """
    Resolves `source` into something cv2.VideoCapture can open.

    - A local path is returned unchanged (after checking it exists).
    - A YouTube URL is resolved via yt-dlp into a direct stream URL,
      preferring H.264 (avc1) or VP9 and explicitly excluding AV1 (av01),
      which OpenCV's build frequently cannot decode.
    """
    is_youtube = (
        source.startswith("http://")
        or source.startswith("https://")
        or "youtube.com" in source
        or "youtu.be" in source
    )

    if not is_youtube:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Local video file '{source}' does not exist.")
        return source

    if config.VERBOSE:
        print(f"Resolving YouTube video stream URL from: {source} ...")
    import yt_dlp

    ydl_opts = {
        "format": "bestvideo[vcodec^=avc1]/bestvideo[vcodec^=vp9]/bestvideo[vcodec!=av01]/best",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=False)

        if "url" in info:
            return info["url"]

        if info.get("requested_formats"):
            for fmt in info["requested_formats"]:
                vcodec = fmt.get("vcodec", "")
                if vcodec != "none" and not vcodec.startswith("av01") and "url" in fmt:
                    return fmt["url"]
            return info["requested_formats"][0]["url"]

        if info.get("formats"):
            for fmt in reversed(info["formats"]):
                vcodec = fmt.get("vcodec", "")
                if vcodec != "none" and not vcodec.startswith("av01") and "url" in fmt:
                    return fmt["url"]
            return info["formats"][-1]["url"]

        raise RuntimeError("Could not find a valid OpenCV-compatible video stream URL.")


class FrameReader:
    """
    Reads frames from a cv2.VideoCapture on a dedicated background thread
    and pushes them into a bounded queue.

    This decouples slow network/decode I/O (especially for remote YouTube
    streams) from optical-flow processing: the analyzer thread never stalls
    inside cap.read(), and a temporary network hiccup doesn't stall capture
    either since it happens off the processing thread.

    If the consumer falls behind and the queue fills up, the oldest queued
    frame is dropped in favor of the newest one -- appropriate for a
    near-real-time analytics pipeline, where staying current matters more
    than processing every single frame.
    """

    def __init__(self, stream_url: str, queue_size: int = None):
        self.cap = cv2.VideoCapture(stream_url)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video stream from '{stream_url}'")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 0 for streams/live sources and some containers; used only for
        # progress reporting, so an unknown total is fine.
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        self._queue = queue.Queue(maxsize=queue_size or config.FRAME_QUEUE_SIZE)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="frame-reader")

    def start(self) -> "FrameReader":
        self._thread.start()
        return self

    def _run(self):
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._queue.put(None)  # sentinel: end of stream
                return
            try:
                self._queue.put(frame, timeout=1.0)
            except queue.Full:
                try:
                    self._queue.get_nowait()  # drop the oldest, keep pace with live input
                except queue.Empty:
                    pass
                self._queue.put(frame)

    def read(self, timeout: float = None):
        """Blocking read of the next frame. Returns None at end of stream or on timeout."""
        try:
            return self._queue.get(timeout=timeout or config.READER_QUEUE_TIMEOUT)
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()
        self.cap.release()
