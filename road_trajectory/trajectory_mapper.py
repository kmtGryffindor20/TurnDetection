"""Dense-optical-flow heading estimation + kinematic-bicycle odometry.

Per analysed frame:

    frame -> resize -> grayscale -> blank above the horizon
          -> Farneback flow -> mean horizontal displacement (OF_x)
          -> rolling average + deadband -> yaw increment (radians)
          -> bicycle model -> (x, y) position

Coordinate frame (top-down, matches the vertical plot):

    +Y = forward (up the page)      +X = right (lateral)
    heading 0 = straight ahead      positive heading = a LEFT turn
    x -= v * sin(heading)           y += v * cos(heading)

so a left-hand bend curves towards -X (left on the plot) and the bike always
travels up the page.
"""

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

import trajectory_config as config


@dataclass(frozen=True)
class TrajectoryState:
    """One analysed frame's contribution to the trajectory."""

    frame_index: int
    time_s: float
    flow_x: float  # smoothed average horizontal flow (pixels / frame)
    yaw: float  # heading change this step (radians, + = left)
    heading: float  # accumulated heading (radians, + = left)
    x: float  # lateral offset, metres (+ = right)
    y: float  # distance travelled, metres (+ = forward)
    speed_kmh: float


class FlowYawEstimator:
    """
    Converts the average horizontal optical flow of the road surface into a
    per-step yaw increment.

    `update(frame)` returns `(flow_x, yaw, flow)` where `flow_x` is the
    smoothed/deadbanded horizontal displacement, `yaw` is the same value in
    radians (positive = left turn, see `config.YAW_SIGN`), and `flow` is the
    dense field (None on the first frame, which only sets the reference image).
    """

    def __init__(self):
        self._history = deque(maxlen=max(1, int(config.SMOOTHING_WINDOW)))
        self._prev_gray = None

    def reset(self):
        self._history.clear()
        self._prev_gray = None

    def _prepare(self, frame):
        """Resize, grayscale, mask the sky; return (gray, roi_bounds)."""
        if config.PROCESS_SIZE is not None:
            frame = cv2.resize(
                frame, config.PROCESS_SIZE, interpolation=cv2.INTER_AREA
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        y0 = int(h * config.ROI_Y[0])
        y1 = int(h * config.ROI_Y[1])
        x0 = int(w * config.ROI_X[0])
        x1 = int(w * config.ROI_X[1])

        # Blank the sky / distant scene: moving clouds and far-away objects
        # would otherwise skew the horizontal average.
        gray[:y0, :] = 0
        return gray, (y0, y1, x0, x1)

    def update(self, frame):
        gray, (y0, y1, x0, x1) = self._prepare(frame)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0, 0.0, None

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, **config.FARNEBACK
        )
        self._prev_gray = gray

        horizontal = flow[y0:y1, x0:x1, 0]
        if horizontal.size == 0:
            return 0.0, 0.0, flow

        stride = max(1, int(config.MEAN_STRIDE))
        raw_dx = float(np.mean(horizontal[::stride, ::stride]))

        # Rolling average: frame-to-frame flow is jittery (engine vibration).
        self._history.append(raw_dx)
        smoothed = float(np.mean(self._history))

        # Deadband: kill the slow drift while the bike is going straight.
        if abs(smoothed) < config.DEADBAND_PX:
            smoothed = 0.0

        yaw = config.YAW_SIGN * config.PIXELS_TO_RADIANS * smoothed
        return smoothed, yaw, flow


class KinematicBicycle:
    """
    Constant-speed kinematic bicycle model integrated per step.

    Heading 0 points forward (+Y); a positive heading is a left turn, and the
    position update moves along the heading.
    """

    def __init__(self, speed_kmh=None):
        self.speed_kmh = config.SPEED_KMH if speed_kmh is None else speed_kmh
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.last_step = 0.0
        self.last_speed_kmh = 0.0

    def step(self, yaw_increment, dt, speed_kmh=None):
        """Advance one analysed step of `dt` seconds -> (x, y, heading)."""
        speed = self.speed_kmh if speed_kmh is None else speed_kmh

        if speed is None or speed <= 0:
            distance = config.FALLBACK_STEP_M
            speed = 0.0
        else:
            distance = (float(speed) / 3.6) * dt

        # Update the heading first, then travel along the new direction.
        self.heading += yaw_increment
        self.x -= distance * math.sin(self.heading)
        self.y += distance * math.cos(self.heading)

        self.last_step = distance
        self.last_speed_kmh = float(speed)
        return self.x, self.y, self.heading


class RoadTrajectoryMapper:
    """
    End-to-end mapper: video frame in, 2D position out.

    Pass the measured speed with every frame (`speed_kmh=`); omitting it falls
    back to the constant in the config. Call `update(frame, frame_index)` once
    per analysed frame and keep the returned `TrajectoryState`; `.path` holds
    every (x, y) point so far.
    """

    def __init__(self, fps, speed_kmh=None):
        self.fps = max(float(fps), 1.0)
        self.flow = FlowYawEstimator()
        self.bicycle = KinematicBicycle(speed_kmh)

        self.path = [(0.0, 0.0)]
        self.states = []
        self.path_length = 0.0
        self.speeds = []
        self._last_index = None

    @property
    def average_speed_kmh(self):
        """Mean of the speeds actually used by the odometry (0 if unknown)."""
        return float(np.mean(self.speeds)) if self.speeds else 0.0

    def update(self, frame, frame_index=None, speed_kmh=None):
        """Analyse one frame -> (TrajectoryState, flow)."""
        if frame_index is None:
            if self._last_index is None:
                frame_index = 0
            else:
                frame_index = self._last_index + config.FRAME_STEP

        # Time covered by this analysed step (FRAME_STEP may skip frames).
        if self._last_index is None:
            dt = config.FRAME_STEP / self.fps
        else:
            dt = max(frame_index - self._last_index, 1) / self.fps
        self._last_index = frame_index

        flow_x, yaw, flow = self.flow.update(frame)
        x, y, heading = self.bicycle.step(yaw, dt, speed_kmh)

        self.path_length += self.bicycle.last_step
        self.path.append((x, y))
        if self.bicycle.last_speed_kmh > 0:
            self.speeds.append(self.bicycle.last_speed_kmh)

        state = TrajectoryState(
            frame_index=frame_index,
            time_s=frame_index / self.fps,
            flow_x=flow_x,
            yaw=yaw,
            heading=heading,
            x=x,
            y=y,
            speed_kmh=self.bicycle.last_speed_kmh,
        )
        self.states.append(state)
        return state, flow
