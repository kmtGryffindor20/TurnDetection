"""Configurable optical-flow based turn detection state machine."""

from collections import deque

import cv2
import numpy as np

import config


class TurnDetector:
    """
    Maintains rolling optical-flow state across frames and reports when a
    turn event starts, is ongoing, or has just completed.

    Call `update(frame)` on every frame you want analyzed. While a
    post-turn clip is being captured elsewhere in the pipeline and you
    don't want a full flow computation, call `sync_gray(frame)` instead so
    `prev_gray` stays current -- otherwise the next real `update()` call
    would diff frames that are seconds apart and misfire the camera-cut
    filter.
    """

    def __init__(self, first_frame):
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.dx_history = deque(maxlen=config.SMOOTHING_WINDOW)

        self.is_turning = False
        self.turn_step_counter = 0
        self.accumulated_dx = 0.0

        self.current_dx = 0.0
        self.current_dy = 0.0

    def sync_gray(self, frame):
        """Advance prev_gray without running flow analysis."""
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _compute_flow(self, gray):
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        h, w = gray.shape
        y0, y1 = int(h * config.ROI_Y[0]), int(h * config.ROI_Y[1])
        x0, x1 = int(w * config.ROI_X[0]), int(w * config.ROI_X[1])
        roi_flow = flow[y0:y1, x0:x1]
        return float(np.mean(roi_flow[..., 0])), float(np.mean(roi_flow[..., 1])), flow

    def update(self, frame):
        """
        Feed one frame into the detector.

        Returns (event, deflection):
          "cut"       - camera-cut noise suppressed, state reset
          "ongoing"   - mid-turn, still accumulating
          "completed" - a turn just ended; deflection is the accumulated dx
          "idle"      - no turn active

        `deflection` is meaningful only for "completed".
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_dx, raw_dy, flow = self._compute_flow(gray)
        self.prev_gray = gray

        # render = draw_flow(gray, flow)
        # render = cv2.resize(render, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
        # cv2.imshow("Optical Flow Preview", render)

        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     cv2.destroyAllWindows()
        #     return "quit", 0.0

        if (
            abs(raw_dx) > config.CAMERA_CUT_DISCONTINUITY
            or abs(raw_dy) > config.CAMERA_CUT_DISCONTINUITY
        ):
            self.is_turning = False
            self.turn_step_counter = 0
            self.dx_history.clear()
            return "cut", 0.0

        self.dx_history.append(raw_dx)
        smooth_dx = float(np.mean(self.dx_history))
        self.current_dx = smooth_dx
        self.current_dy = raw_dy

        is_valid_turn_vector = abs(smooth_dx) > config.TURN_THRESHOLD and abs(
            smooth_dx
        ) > config.RATIO_DOMINANCE * abs(raw_dy)

        if is_valid_turn_vector:
            if not self.is_turning:
                self.is_turning = True
                self.turn_step_counter = 0
                self.accumulated_dx = 0.0
            self.turn_step_counter += 1
            self.accumulated_dx += smooth_dx
            return "ongoing", 0.0

        if self.is_turning:
            self.is_turning = False
            completed_duration = self.turn_step_counter
            completed_dx = self.accumulated_dx
            self.turn_step_counter = 0
            if completed_duration >= config.MIN_TURN_DURATION:
                return "completed", completed_dx

        return "idle", 0.0
