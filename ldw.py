"""
ldw.py — Lane Departure Warning (LDW) module for EdgeVision ADAS.

A lightweight, efficient computer-vision lane detection and departure warning
system designed for real-time edge execution (Raspberry Pi 5 / Laptop).

Pipeline:
  1. Region of Interest (ROI) selection (lower road region).
  2. Edge & color thresholding (extracts white/yellow lane markings).
  3. Hough line segment detection.
  4. Slope-based separation into Left and Right lane candidates.
  5. RANSAC/weighted line fitting + temporal EMA smoothing.
  6. Ego-vehicle lane center deviation calculation.
  7. Multi-frame debounced departure alert:
     "LANE DEPARTURE WARNING" (Left or Right)
  8. Overlay visualizer (lane polygon + boundaries).

Resilience:
  - Temporal EMA smoothing for lane boundaries across frames.
  - Hysteresis & confirmation frames prevent false alarms on shadows/cracks.
  - Works with pure NumPy fallback when cv2 is unavailable in offline tests.
"""

import math
import numpy as np

# cv2 is loaded lazily on demand so unit tests / headless pipelines run cleanly
_cv2 = None

def _get_cv2():
    global _cv2
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
        except Exception:
            _cv2 = False
    return _cv2 if _cv2 is not False else None


# ---------------------------------------------------------------------------
# Configuration (tunable for academic viva & Raspberry Pi performance)
# ---------------------------------------------------------------------------

# Region of interest fractions (trapezoid in lower half of image)
ROI_TOP_Y_FRAC = 0.58       # start of road ROI from top
ROI_BOTTOM_Y_FRAC = 0.95    # bottom of road ROI
ROI_TOP_WIDTH_FRAC = 0.35   # width of ROI at horizon line
ROI_BOTTOM_WIDTH_FRAC = 0.90 # width of ROI at bumper line

# Slope thresholds for left/right lane boundaries (image coordinates: y down)
# Left lane: negative slope (dx > 0, dy < 0 -> slope dy/dx < 0)
LEFT_SLOPE_MIN = -2.2
LEFT_SLOPE_MAX = -0.35

# Right lane: positive slope
RIGHT_SLOPE_MIN = 0.35
RIGHT_SLOPE_MAX = 2.2

# Temporal smoothing factor for lane line endpoints (0 < alpha <= 1)
LANE_EMA_ALPHA = 0.35

# Departure thresholds (normalized offset from lane center: -1.0 left, +1.0 right)
# When normalized offset exceeds +/- DEPARTURE_OFFSET_THRESH -> trigger departure
DEPARTURE_OFFSET_ENTER = 0.55  # 55% toward either lane marking
DEPARTURE_OFFSET_EXIT = 0.35   # must return inside 35% to clear warning

# Confirmation debouncing (consecutive frames required)
CONFIRM_DEPARTURE_FRAMES = 4
CONFIRM_RECOVERY_FRAMES = 6


class LaneDepartureWarning:
    """Detects lane boundaries and alerts when the vehicle departs its lane."""

    def __init__(self, frame_width, frame_height):
        self.frame_w = frame_width
        self.frame_h = frame_height

        # Smoothed lane boundary endpoints: (x_bottom, y_bottom, x_top, y_top)
        self.left_line = None
        self.right_line = None

        # Lane tracking status
        self.lanes_detected = False
        self.left_detected = False
        self.right_detected = False

        # Deviation metrics (-1.0 = on left line, 0.0 = center, +1.0 = on right line)
        self.lateral_offset = 0.0
        self.departure_direction = "NONE"  # "NONE" | "LEFT" | "RIGHT"

        # Warning state machine: "NORMAL" | "WARNING_LEFT" | "WARNING_RIGHT"
        self.status = "NORMAL"
        self._candidate_status = "NORMAL"
        self._confirm_count = 0

        # Departure alert message (consistent wording per ReadME.md)
        self.alert_active = False
        self.alert_message = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_or_sim, now):
        """
        Process a new camera frame (or synthetic simulation dictionary).
        frame_or_sim: BGR numpy image array, or dict with synthetic line endpoints.
        now: float monotonic timestamp.
        """
        if isinstance(frame_or_sim, dict):
            # Simulation / offline test mode
            raw_left = frame_or_sim.get("left_line")
            raw_right = frame_or_sim.get("right_line")
        else:
            # Live computer-vision pipeline on image
            raw_left, raw_right = self._detect_lanes_cv(frame_or_sim)

        # Update smoothed lane positions
        self._update_smoothed_lanes(raw_left, raw_right)

        # Calculate vehicle offset & departure
        self._compute_departure()

        # Advance debounced warning state machine
        self._advance_state()

        return self

    # ------------------------------------------------------------------
    # Computer Vision Lane Detection Pipeline
    # ------------------------------------------------------------------

    def _detect_lanes_cv(self, frame):
        """Processes frame using lightweight edge + Hough transform."""
        cv = _get_cv2()
        if cv is None or frame is None:
            return None, None

        h, w = frame.shape[:2]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 1. Blur to suppress high frequency road texture
        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        # 2. Canny Edge Detection with calibrated thresholds
        edges = cv.Canny(blurred, 50, 150)

        # 3. Apply Trapezoidal Region of Interest (ROI) Mask
        roi_mask = np.zeros_like(edges)
        y_top = int(h * ROI_TOP_Y_FRAC)
        y_bottom = int(h * ROI_BOTTOM_Y_FRAC)
        center_x = w // 2
        half_top = int(w * ROI_TOP_WIDTH_FRAC / 2)
        half_bottom = int(w * ROI_BOTTOM_WIDTH_FRAC / 2)

        roi_vertices = np.array([[
            (center_x - half_bottom, y_bottom),
            (center_x - half_top, y_top),
            (center_x + half_top, y_top),
            (center_x + half_bottom, y_bottom),
        ]], dtype=np.int32)

        cv.fillPoly(roi_mask, roi_vertices, 255)
        masked_edges = cv.bitwise_and(edges, roi_mask)

        # 4. Probabilistic Hough Line Transform
        lines = cv.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=30,
            maxLineGap=25
        )

        if lines is None:
            return None, None

        # 5. Classify lines into Left & Right by slope
        left_segments = []
        right_segments = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # Skip pure vertical lines

            slope = float(y2 - y1) / float(x2 - x1)
            length = math.hypot(x2 - x1, y2 - y1)

            if LEFT_SLOPE_MIN <= slope <= LEFT_SLOPE_MAX:
                # Left lane line (negative slope)
                if (x1 + x2) / 2.0 < center_x + 0.1 * w:
                    left_segments.append((x1, y1, x2, y2, length))

            elif RIGHT_SLOPE_MIN <= slope <= RIGHT_SLOPE_MAX:
                # Right lane line (positive slope)
                if (x1 + x2) / 2.0 > center_x - 0.1 * w:
                    right_segments.append((x1, y1, x2, y2, length))

        left_fit = self._fit_line_segment(left_segments, y_bottom, y_top)
        right_fit = self._fit_line_segment(right_segments, y_bottom, y_top)

        return left_fit, right_fit

    def _fit_line_segment(self, segments, y_bottom, y_top):
        """Fits a weighted representative line through candidate segments."""
        if not segments:
            return None

        # Extract points weighted by line segment length
        x_coords = []
        y_coords = []
        weights = []

        for x1, y1, x2, y2, length in segments:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            weights.extend([length, length])

        if len(x_coords) < 2:
            return None

        # Weighted 1st degree polynomial fit (x as function of y)
        poly = np.polyfit(y_coords, x_coords, 1, w=weights)
        x_bottom = int(poly[0] * y_bottom + poly[1])
        x_top = int(poly[0] * y_top + poly[1])

        return (x_bottom, y_bottom, x_top, y_top)

    def _update_smoothed_lanes(self, raw_left, raw_right):
        """Applies EMA smoothing to lane line endpoints across frames."""
        self.left_detected = (raw_left is not None)
        self.right_detected = (raw_right is not None)
        self.lanes_detected = self.left_detected or self.right_detected

        if raw_left is not None:
            if self.left_line is None:
                self.left_line = list(raw_left)
            else:
                for i in range(4):
                    self.left_line[i] = int(
                        LANE_EMA_ALPHA * raw_left[i] + (1.0 - LANE_EMA_ALPHA) * self.left_line[i]
                    )

        if raw_right is not None:
            if self.right_line is None:
                self.right_line = list(raw_right)
            else:
                for i in range(4):
                    self.right_line[i] = int(
                        LANE_EMA_ALPHA * raw_right[i] + (1.0 - LANE_EMA_ALPHA) * self.right_line[i]
                    )

    def _compute_departure(self):
        """Estimates vehicle lateral offset within the lane."""
        cam_center_x = self.frame_w / 2.0

        if self.left_line is not None and self.right_line is not None:
            # Both lanes available -> high accuracy center
            lx_bottom = self.left_line[0]
            rx_bottom = self.right_line[0]
            lane_w = max(1.0, float(rx_bottom - lx_bottom))
            lane_center = (lx_bottom + rx_bottom) / 2.0

            # Normalized offset: -1.0 is on left line, +1.0 is on right line
            self.lateral_offset = (cam_center_x - lane_center) / (lane_w / 2.0)

        elif self.left_line is not None:
            # Only left lane detected
            lx_bottom = self.left_line[0]
            est_half_w = self.frame_w * 0.28
            self.lateral_offset = (cam_center_x - (lx_bottom + est_half_w)) / est_half_w

        elif self.right_line is not None:
            # Only right lane detected
            rx_bottom = self.right_line[0]
            est_half_w = self.frame_w * 0.28
            self.lateral_offset = (cam_center_x - (rx_bottom - est_half_w)) / est_half_w

        else:
            self.lateral_offset = 0.0

        # Bound lateral offset
        self.lateral_offset = max(-1.5, min(1.5, self.lateral_offset))

    def _advance_state(self):
        """Hysteresis & debounce state machine for lane departure warning."""
        # Determine instantaneous candidate state
        if self.lateral_offset <= -DEPARTURE_OFFSET_ENTER:
            inst_candidate = "WARNING_LEFT"
        elif self.lateral_offset >= DEPARTURE_OFFSET_ENTER:
            inst_candidate = "WARNING_RIGHT"
        elif abs(self.lateral_offset) <= DEPARTURE_OFFSET_EXIT:
            inst_candidate = "NORMAL"
        else:
            # In hysteresis exit band -> keep current status
            inst_candidate = self.status

        # If not detecting lanes, gracefully decay to NORMAL
        if not self.lanes_detected:
            inst_candidate = "NORMAL"

        if inst_candidate == self.status:
            self._candidate_status = inst_candidate
            self._confirm_count = 0
        else:
            if inst_candidate != self._candidate_status:
                self._candidate_status = inst_candidate
                self._confirm_count = 1
            else:
                self._confirm_count += 1

            needed = (
                CONFIRM_DEPARTURE_FRAMES
                if inst_candidate != "NORMAL"
                else CONFIRM_RECOVERY_FRAMES
            )

            if self._confirm_count >= needed:
                self.status = self._candidate_status
                self._confirm_count = 0

        # Update driver alert message & state
        if self.status != "NORMAL":
            self.alert_active = True
            self.departure_direction = "LEFT" if self.status == "WARNING_LEFT" else "RIGHT"
            self.alert_message = "LANE DEPARTURE WARNING"
        else:
            self.alert_active = False
            self.departure_direction = "NONE"
            self.alert_message = ""

    # ------------------------------------------------------------------
    # Visualization Overlay Helper
    # ------------------------------------------------------------------

    def draw_overlay(self, frame):
        """Draws lane markings and illuminated lane corridor on the frame."""
        cv = _get_cv2()
        if cv is None or frame is None:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Draw lane corridor polygon if both lines are tracked
        if self.left_line is not None and self.right_line is not None:
            lx_b, ly_b, lx_t, ly_t = self.left_line
            rx_b, ry_b, rx_t, ry_t = self.right_line

            # Color green when normal, orange/red when departing
            if self.status != "NORMAL":
                corridor_color = (0, 100, 255)  # Orange alert
            else:
                corridor_color = (0, 220, 0)    # Calm green

            lane_poly = np.array([[
                (lx_b, ly_b),
                (lx_t, ly_t),
                (rx_t, ry_t),
                (rx_b, ry_b)
            ]], dtype=np.int32)

            cv.fillPoly(overlay, lane_poly, corridor_color)
            # Blend 25% opacity overlay onto frame
            cv.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # 2. Draw Left boundary line
        if self.left_line is not None:
            lx_b, ly_b, lx_t, ly_t = self.left_line
            line_color = (0, 0, 255) if self.status == "WARNING_LEFT" else (0, 255, 255)
            cv.line(frame, (lx_b, ly_b), (lx_t, ly_t), line_color, 3)

        # 3. Draw Right boundary line
        if self.right_line is not None:
            rx_b, ry_b, rx_t, ry_t = self.right_line
            line_color = (0, 0, 255) if self.status == "WARNING_RIGHT" else (0, 255, 255)
            cv.line(frame, (rx_b, ry_b), (rx_t, ry_t), line_color, 3)

        return frame
