"""Speedometer readout recognition (EasyOCR).

The readout sits in the lower-middle of the frame. EasyOCR's learned text
detector returns nothing when there is no text, which is what keeps roadside
scenery from being read as a speed; the largest text box found inside the
configured ROI is taken as the reading.

Mirrors the rest of the pipeline (`TurnDetector`, `FrameReader`, ...):
construct once, then call `read(frame)` per frame.
"""
import cv2
import numpy as np

import config


class SpeedRecognizer:
    """
    Reads the speedometer value from a frame.

    `read(frame)` returns `(speed, bbox)`:
      - `speed` is the recognised value as an int (km/h), or None.
      - `bbox` is the detected text box in full-frame coordinates
        (4x2 float array), or None.
    """

    def __init__(self):
        self.enabled = config.SPEED_ENABLED
        self._reader = None

    @property
    def reader(self):
        # Loaded lazily so importing this module stays cheap and the models /
        # GPU are only touched once a frame is actually read.
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(
                list(config.SPEED_LANGS),
                gpu=config.SPEED_GPU,
                model_storage_directory=config.SPEED_MODEL_DIR,
                user_network_directory=config.SPEED_MODEL_DIR,
            )
        return self._reader

    def _roi(self, frame):
        h, w = frame.shape[:2]
        y0, y1 = int(h * config.SPEED_ROI_Y[0]), int(h * config.SPEED_ROI_Y[1])
        x0, x1 = int(w * config.SPEED_ROI_X[0]), int(w * config.SPEED_ROI_X[1])
        return frame[y0:y1, x0:x1], x0, y0

    @staticmethod
    def _largest(results):
        """Largest text box in the OCR results: (bbox, text) or (None, None)."""
        largest_bbox, text, largest_area = None, None, 0.0
        for bbox, candidate, _prob in results:
            area = (bbox[2][0] - bbox[0][0]) * (bbox[2][1] - bbox[0][1])
            if largest_bbox is None or area > largest_area:
                largest_bbox, text, largest_area = bbox, candidate, area
        return largest_bbox, text

    @staticmethod
    def _to_int(text):
        digits = "".join(ch for ch in (text or "") if ch.isdigit())
        return int(digits) if digits else None

    def read(self, frame):
        """Recognise the speedometer reading in `frame` -> (speed, bbox)."""
        if not self.enabled:
            return None, None

        roi, offset_x, offset_y = self._roi(frame)
        if roi.size == 0:
            return None, None

        largest_bbox, text = self._largest(self.reader.readtext(roi))
        if largest_bbox is None:
            return None, None

        bbox = np.array(largest_bbox, dtype="float32")
        bbox[:, 0] += offset_x
        bbox[:, 1] += offset_y
        return self._to_int(text), bbox


if __name__ == "__main__":
    # Standalone preview: recognise the reading live and draw its box.
    recognizer = SpeedRecognizer()
    cap = cv2.VideoCapture(config.INPUT_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = 1

    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            speed, bbox = recognizer.read(frame)
            if bbox is not None:
                p = bbox.astype(int)
                cv2.rectangle(frame, (p[0][0], p[0][1]), (p[2][0], p[2][1]), (0, 255, 0), 2)
            cv2.putText(
                frame, f"Speed: {speed}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.imshow(
                "Speed Recognition",
                cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)),
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        idx += 1

    cap.release()
    cv2.destroyAllWindows()
