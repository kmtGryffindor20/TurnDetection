"""
Speedometer OCR for the turn-detection pipeline (EasyOCR, GPU).

On turn completion the lower part of the frame is OCR'd. If any numeric text is
found, the largest numeric token is taken as the motorcycle's speed and the turn
is saved; otherwise the turn is discarded.

There is deliberately no plausibility (0-199) or confidence check: EasyOCR ships
a learned text *detector* (CRAFT) in front of the recognizer, and it returns no
boxes at all when there is no text, which is what stops scenery (road, grass)
from being read as a speed. Only the digits themselves are trusted as the value.
"""
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2

import config

try:
    import easyocr

    _EASYOCR_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import guard for clear messaging
    easyocr = None
    _EASYOCR_IMPORT_ERROR = exc


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class SpeedReading:
    """Result of one OCR pass over the speedometer region."""

    text: str = ""
    speed: Optional[int] = None
    bbox: Optional[tuple] = None  # (x0, y0, x1, y1) in full-frame coordinates
    area: float = 0.0
    detected: bool = False

    def __str__(self) -> str:
        if self.detected:
            return f"{self.speed} km/h ('{self.text}')"
        return "no numeric text"


class SpeedReader:
    """
    Crops the configured OCR region and reads the largest numeric token from it
    with EasyOCR. A reading counts as `detected` as soon as any digits are found
    (no range or confidence filtering).
    """

    def __init__(self):
        self.available = False
        self.unavailable_reason = ""
        self._reader = None
        self._digit_re = re.compile(r"\d+")

        if _EASYOCR_IMPORT_ERROR is not None:
            self.unavailable_reason = f"easyocr import failed: {_EASYOCR_IMPORT_ERROR}"
            return

        model_dir = config.EASYOCR_MODEL_DIR
        if model_dir and not os.path.isabs(model_dir):
            model_dir = os.path.join(PROJECT_ROOT, model_dir)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        try:
            self._reader = easyocr.Reader(
                list(config.EASYOCR_LANGS),
                gpu=config.EASYOCR_GPU,
                model_storage_directory=model_dir or None,
                download_enabled=True,
                verbose=False,
            )
            self.available = True
            print(
                f"[OCR] EasyOCR ready (langs={list(config.EASYOCR_LANGS)}, "
                f"gpu={config.EASYOCR_GPU})"
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            self.unavailable_reason = f"EasyOCR could not be initialised: {exc}"

    def _crop(self, frame):
        """Returns (roi, x_offset, y_offset) for the configured OCR region."""
        h, w = frame.shape[:2]
        y0, y1 = int(h * config.OCR_ROI_Y[0]), int(h * config.OCR_ROI_Y[1])
        x0, x1 = int(w * config.OCR_ROI_X[0]), int(w * config.OCR_ROI_X[1])
        return frame[y0:y1, x0:x1], x0, y0

    def read(self, frame) -> SpeedReading:
        if not self.available:
            return SpeedReading()

        roi, x_off, y_off = self._crop(frame)
        if roi.size == 0:
            return SpeedReading()

        try:
            results = self._reader.readtext(
                roi,
                allowlist=config.OCR_DIGIT_ALLOWLIST,
                detail=1,
                paragraph=False,
                mag_ratio=config.OCR_DETECT_MAG_RATIO,
                low_text=config.OCR_LOW_TEXT,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[OCR] EasyOCR pass failed: {exc}")
            return SpeedReading()

        best = None
        for item in results:
            bbox, text = item[0], item[1]
            digits = "".join(self._digit_re.findall(text or ""))
            if not digits:
                continue
            rect = self._box_rect(bbox)
            if rect is None:
                continue
            area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            if best is None or area > best[0]:
                best = (area, digits, rect)
        if best is None:
            return SpeedReading()

        area, digits, rect = best
        full_bbox = (
            rect[0] + x_off,
            rect[1] + y_off,
            rect[2] + x_off,
            rect[3] + y_off,
        )
        return SpeedReading(
            text=digits, speed=int(digits), bbox=full_bbox, area=area, detected=True
        )

    @staticmethod
    def _box_rect(bbox):
        """Integer (x0, y0, x1, y1) for an EasyOCR 4-point box, or None."""
        try:
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
        except (TypeError, IndexError, ValueError):
            return None
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


class TurnGate:
    """
    Callable adapter handed to `TurnDetector` as its `completion_validator`.

    OCRs the completion frame plus up to `samples - 1` recent frames from the
    pre-turn buffer. The turn is accepted as soon as one of them yields numeric
    text -- the readout is only legible on a fraction of frames at 640x360, so
    sampling several is what makes the gate usable. The largest numeric token
    found is cached on `last` so the pipeline can attach the speed to the clip.
    """

    def __init__(
        self,
        reader: SpeedReader,
        frame_provider: Optional[Callable[[int], List]] = None,
        samples: Optional[int] = None,
    ):
        self.reader = reader
        self.frame_provider = frame_provider
        self.samples = max(1, samples or config.OCR_GATE_SAMPLES)
        self.last = SpeedReading()

    def __call__(self, frame) -> bool:
        frames = [frame]
        if self.frame_provider is not None and self.samples > 1:
            frames.extend(self.frame_provider(self.samples - 1))

        best = SpeedReading()
        for candidate in frames:
            reading = self.reader.read(candidate)
            if reading.detected and reading.area > best.area:
                best = reading
        self.last = best
        return best.detected

    @property
    def speed(self) -> Optional[int]:
        return self.last.speed if self.last.detected else None
