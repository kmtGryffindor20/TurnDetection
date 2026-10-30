"""Trajectory visualization: final matplotlib plot, live preview, path inset.

Everything is drawn top-down with the bike travelling UP the page: +Y (forward)
is vertical, +X (right) is horizontal.
"""

import math
import os

import cv2
import numpy as np

import trajectory_config as config


def plot_trajectory(
    path,
    output_path=None,
    show=None,
    title=None,
    distance=None,
    heading_deg=None,
    average_speed=None,
):
    """
    Draws the 2D path vertically (forward = up) with an equal aspect ratio, so
    a 90-degree turn cannot look like a shallow curve, and optionally saves it.

    `path` is a sequence of (x, y) points; x is lateral (+ = right), y is
    forward (+ = up the page).
    """
    show = config.SHOW_PLOT if show is None else show
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or len(points) < 2:
        print("Trajectory is too short to plot.")
        return None

    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
    ax.plot(points[:, 0], points[:, 1], color="#1f77b4", linewidth=1.8)
    ax.plot(
        points[0, 0], points[0, 1], "o", color="#2ca02c", markersize=9, label="start"
    )
    ax.plot(
        points[-1, 0], points[-1, 1], "s", color="#d62728", markersize=8, label="end"
    )

    # Travel direction at the start of the clip.
    ax.annotate(
        "",
        xy=(points[1, 0], points[1, 1]),
        xytext=(points[0, 0], points[0, 1]),
        arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=1.5),
    )

    ax.set_aspect("equal")  # never auto-scale X and Y independently
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("lateral X (m)   + = right")
    ax.set_ylabel("forward Y (m)   + = up")
    ax.legend(loc="best")

    # A mostly-straight path would otherwise collapse into a hairline; keep a
    # minimum plotting width (the aspect stays 1:1, this only adds margin).
    xs, ys = points[:, 0], points[:, 1]
    span_x = max(float(np.ptp(xs)), 1e-6)
    span_y = max(float(np.ptp(ys)), 1e-6)
    pad_y = span_y * 0.04 + 1.0
    half_width = max(span_x * 0.65, span_y * 0.10)
    mid_x = float((xs.min() + xs.max()) / 2.0)
    ax.set_xlim(mid_x - half_width, mid_x + half_width)
    ax.set_ylim(float(ys.min()) - pad_y, float(ys.max()) + pad_y)

    notes = []
    if distance is not None:
        notes.append(f"distance {distance:.0f} m")
    if heading_deg is not None:
        notes.append(f"net heading {heading_deg:+.1f} deg")
    if average_speed is not None:
        notes.append(f"avg speed {average_speed:.1f} km/h")

    heading_text = title or "Estimated road trajectory"
    if notes:
        heading_text = f"{heading_text}\n" + "   |   ".join(notes)
    ax.set_title(heading_text, fontsize=11)

    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot written: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def draw_trajectory_inset(frame, path, size=None, margin=10):
    """
    Draws a scaled copy of the path into the bottom-right corner of `frame`
    (in place), forward pointing up. Math +Y is up but image +Y is down, so the
    drawing is flipped vertically.
    """
    points = np.asarray(path, dtype=float)
    if len(points) < 2:
        return frame

    width, height = size or config.INSET_SIZE
    h, w = frame.shape[:2]
    width = min(width, max(w - 2 * margin, 40))
    height = min(height, max(h - 2 * margin, 40))
    x0 = w - width - margin
    y0 = h - height - margin

    panel = frame[y0 : y0 + height, x0 : x0 + width]
    backdrop = panel.copy()
    cv2.rectangle(backdrop, (0, 0), (width - 1, height - 1), (0, 0, 0), -1)
    cv2.addWeighted(backdrop, 0.55, panel, 0.45, 0, panel)

    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1e-6)
    pad = 14
    scale = (min(width, height) - 2 * pad) / span
    cx = float((points[:, 0].min() + points[:, 0].max()) / 2.0)
    cy = float((points[:, 1].min() + points[:, 1].max()) / 2.0)

    px = (points[:, 0] - cx) * scale + width / 2.0 + x0
    py = height / 2.0 - (points[:, 1] - cy) * scale + y0
    pts = np.stack([px, py], axis=1).astype(np.int32).reshape(-1, 1, 2)

    cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
    cv2.circle(frame, tuple(pts[-1, 0]), 4, (0, 0, 255), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + width - 1, y0 + height - 1), (90, 90, 90), 1)
    return frame


class TrajectoryPreview:
    """
    Live OpenCV window: the frame with the ROI line, a HUD, the recognised
    speedometer box and a path inset.

    Like the pipeline's PreviewWindow, this must be driven from the main
    thread (OpenCV HighGUI requirement).
    """

    def __init__(self, name: str = "Road Trajectory Mapper"):
        self.name = name
        self.enabled = config.SHOW_PREVIEW
        if self.enabled:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.name, config.PREVIEW_WIDTH, config.PREVIEW_HEIGHT
            )

    def render(self, frame, state, path, speed_bbox=None):
        """Builds the annotated preview image (no windowing)."""
        display = cv2.resize(
            frame, (config.PREVIEW_WIDTH, config.PREVIEW_HEIGHT)
        )

        # Horizon line: everything above it is ignored by the flow average.
        y0 = int(display.shape[0] * config.ROI_Y[0])
        cv2.line(display, (0, y0), (display.shape[1], y0), (0, 200, 255), 2)

        # Recognised speedometer readout, scaled from full-frame coordinates.
        if speed_bbox is not None:
            sx = display.shape[1] / frame.shape[1]
            sy = display.shape[0] / frame.shape[0]
            box = np.asarray(speed_bbox, dtype=float).reshape(-1, 2)
            box[:, 0] *= sx
            box[:, 1] *= sy
            box = box.astype(int)
            cv2.rectangle(display, tuple(box[0]), tuple(box[2]), (0, 255, 0), 2)

        if state is not None:
            heading_deg = math.degrees(state.heading)
            lines = [
                f"Frame: {state.frame_index}   t: {state.time_s:6.2f}s",
                f"Heading: {heading_deg:+7.2f} deg   Speed: {state.speed_kmh:5.1f} km/h",
                f"OF_x: {state.flow_x:+6.2f} px   Yaw: {math.degrees(state.yaw):+6.3f} deg",
            ]
            overlay = display.copy()
            cv2.rectangle(overlay, (8, 8), (440, 95), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
            for i, text in enumerate(lines):
                cv2.putText(
                    display,
                    text,
                    (16, 32 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                )

        draw_trajectory_inset(display, path)
        return display

    def show(self, frame, state, path, speed_bbox=None) -> bool:
        """Renders one frame. Returns False if the user pressed 'q'."""
        if not self.enabled:
            return True
        cv2.imshow(self.name, self.render(frame, state, path, speed_bbox))
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self):
        if self.enabled:
            cv2.destroyAllWindows()
