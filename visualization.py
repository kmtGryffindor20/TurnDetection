"""HUD overlay rendering and live preview display."""
import cv2

import config


def draw_visual_telemetry(frame, dx, dy, is_turning):
    """Draws a motion-vector arrow and status/telemetry HUD onto a copy of frame."""
    display = frame.copy()
    h, w = display.shape[:2]

    center_x, center_y = w // 2, h // 2
    end_x = int(center_x + dx * 25)
    end_y = int(center_y + dy * 25)

    cv2.arrowedLine(display, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3, tipLength=0.3)
    cv2.circle(display, (center_x, center_y), 5, (255, 0, 0), -1)

    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (320, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    status_color = (0, 0, 255) if is_turning else (0, 255, 0)
    status_text = "STATUS: TURN DETECTED" if is_turning else "STATUS: CRUISING"

    cv2.putText(display, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
    cv2.putText(display, f"Horiz Vector (dx): {dx:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(display, f"Vert Vector (dy):   {dy:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return display


class PreviewWindow:
    """
    Thin wrapper around an OpenCV HighGUI window.

    OpenCV's imshow/waitKey must run on the main thread, so this is driven
    only from the main loop -- never from a worker thread -- even though
    frame capture and clip encoding run in the background.
    """

    def __init__(self, name: str = "20s Clip 3-Class Turn Dataset Generator"):
        self.name = name
        self.enabled = config.SHOW_PREVIEW
        if self.enabled:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.name, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)

    def show(self, frame, dx, dy, is_turning) -> bool:
        """Renders one frame. Returns False if the user pressed 'q' to stop."""
        if not self.enabled:
            return True
        render = draw_visual_telemetry(frame, dx, dy, is_turning)
        render = cv2.resize(render, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
        cv2.imshow(self.name, render)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        return True

    def close(self):
        if self.enabled:
            cv2.destroyAllWindows()
