"""
ipm_distance.py — Inverse Perspective Mapping (IPM) ground-plane distance.

Replaces the previous pinhole ground-plane estimator, which produced "metric"
numbers from *guessed* camera parameters (cam_height / pitch / fov). This
version is calibrated from four measured points on the road, so the metres it
reports are traceable back to a tape measure.

Method
------
  1. You mark four known points on flat ground in front of the camera and note
     their pixel coordinates in the source frame  (IPM_SRC_POINTS).
  2. You measure the real-world width between the left/right points and the
     real-world depth between the near/far points.
  3. cv2.getPerspectiveTransform() builds a 3x3 homography ONCE, mapping that
     source quad onto a rectangle of IPM_DST_SIZE pixels.
  4. For each detection we take only the bottom-centre of its bounding box
     (the tyre/ground contact point) and push that ONE point through
     cv2.perspectiveTransform(). We never warp the frame just to read a
     distance — that would cost a full-frame resample per frame, which the
     Raspberry Pi 3B cannot afford.
  5. The warped point's pixel position is converted to metres with a
     pixels-per-metre value computed once at calibration time.

Geometry / axis convention
--------------------------
  Source quad, clockwise from the top-left of the road region:

      (0) far-left  --- (1) far-right        <-- the "far" edge
          |                |
      (3) near-left --- (2) near-right       <-- the "near" edge

  Destination rectangle (IPM_DST_SIZE), same ordering:

      (0,0) ------------ (W,0)
        |                    |
      (0,H) ------------ (W,H)

  so the TOP of the bird's-eye image is the far edge and the BOTTOM is the
  edge nearest the camera.

  x_lateral_m : metres from the centreline, positive = right
  z_ahead_m   : metres ahead of the camera, 0 at the near edge,
                IPM_REAL_WORLD_HEIGHT_M at the far edge

Choosing IPM_DST_SIZE
---------------------
  Pick the destination size so that both axes share the same pixels-per-metre:

      IPM_DST_SIZE[0] / IPM_REAL_WORLD_WIDTH_M  ==  IPM_DST_SIZE[1] / IPM_REAL_WORLD_HEIGHT_M

  Example: a 4.0 m wide x 10.0 m deep patch at IPM_DST_SIZE = (400, 1000)
  gives 100 px/m on both axes. Square pixels mean the bird's-eye view is not
  stretched, and every metre in it is exactly `ppm` pixels.
"""

import math  # only used by the commented-out previous-method reference at the bottom

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Calibration defaults
#
# !! REPLACE THESE WITH YOUR OWN MEASURED VALUES BEFORE TRUSTING ANY NUMBER !!
# See the "HOW TO CALIBRATE IPM" section in the project notes. The defaults
# below are a plausible road trapezoid for a 640x480 frame so the program
# still runs before you calibrate; they are NOT measurements.
# ---------------------------------------------------------------------------

#: 4 points on the road in the SOURCE frame, clockwise from the top-left:
#: (far-left, far-right, near-right, near-left). Must match the real video
#: resolution — measure them on a frame of that exact size.
DEFAULT_SRC_POINTS = [(224, 236), (416, 236), (512, 430), (128, 430)]

#: Real-world width in metres between the left and right points.
DEFAULT_REAL_WORLD_WIDTH_M = 4.0

#: Real-world depth in metres between the far edge and the near edge.
DEFAULT_REAL_WORLD_HEIGHT_M = 10.0

#: Size of the top-down image in pixels. Chosen so both axes are equal
#: pixels-per-metre (see the module docstring): 400/4.0 == 1000/10.0 == 100.
DEFAULT_DST_SIZE = (400, 1000)

#: Sanity limits for the warped point. A point that transforms far outside the
#: destination rectangle is being projected from outside the calibrated road
#: patch (behind the camera, above the horizon, or far off to one side) and is
#: reported as (None, None) rather than as a nonsense distance.
MAX_LATERAL_M = 50.0
MIN_DISTANCE_M = 0.0


class IPMDistanceEstimator:
    """
    Calibrated 4-point IPM distance estimator.

    Same public API as the previous estimator, so fcw.py and pipeline.py keep
    working unchanged:

        estimate_distance(x1, y1, x2, y2) -> (z_ahead_m, x_lateral_m) | (None, None)
        project_to_ground(u, v)          -> (z_ahead_m, x_lateral_m) | (None, None)
        project_to_image(x_m, z_m)       -> (u, v) | (None, None)
        get_horizon_y()                  -> int

    The old pinhole implementation is kept at the bottom of this file as a
    commented reference so you can A/B or revert. See the note there.
    """

    def __init__(self, frame_width=640, frame_height=480,
                 src_points=None,
                 real_world_width_m=DEFAULT_REAL_WORLD_WIDTH_M,
                 real_world_height_m=DEFAULT_REAL_WORLD_HEIGHT_M,
                 dst_size=DEFAULT_DST_SIZE,
                 cam_height_m=None, pitch_angle_deg=None, fov_v_deg=None):
        """
        Parameters:
          frame_width/frame_height: Source frame size in pixels. Used for the
              horizon estimate and validation only; the homography itself is
              defined purely by the four source points.
          src_points: 4 (x, y) pairs on the road, clockwise from top-left.
          real_world_width_m: Metres between the left and right points.
          real_world_height_m: Metres between the near and far points.
          dst_size: (width, height) of the top-down image in pixels.

        cam_height_m / pitch_angle_deg / fov_v_deg are accepted and IGNORED.
        They belonged to the previous pinhole estimator; a measured homography
        does not need them. They are kept in the signature so existing callers
        (and the cockpit settings API) do not crash.
        """
        self.frame_w = float(frame_width)
        self.frame_h = float(frame_height)

        points = src_points if src_points is not None else DEFAULT_SRC_POINTS
        self.src_points = [(float(x), float(y)) for x, y in points]
        if len(self.src_points) != 4:
            raise ValueError("IPM_SRC_POINTS needs exactly 4 (x, y) pairs, got {0}".format(
                len(self.src_points)))

        self.real_w = float(real_world_width_m)
        self.real_h = float(real_world_height_m)
        if self.real_w <= 0 or self.real_h <= 0:
            raise ValueError("IPM real-world width/height must be positive "
                             "(got {0} x {1})".format(self.real_w, self.real_h))

        self.dst_w = int(dst_size[0])
        self.dst_h = int(dst_size[1])
        if self.dst_w <= 0 or self.dst_h <= 0:
            raise ValueError("IPM_DST_SIZE must be positive, got {0}".format((self.dst_w, self.dst_h)))

        # --- Computed ONCE, not per frame ---------------------------------
        self._src = np.array(self.src_points, dtype=np.float32)
        # Clockwise from top-left, matching the source ordering.
        self._dst = np.array(
            [[0, 0], [self.dst_w, 0], [self.dst_w, self.dst_h], [0, self.dst_h]],
            dtype=np.float32,
        )
        self.matrix = cv2.getPerspectiveTransform(self._src, self._dst)
        self.inverse_matrix = _invert_homography(self.matrix)

        # Pixels per metre, computed once. The two axes are allowed to differ
        # but IPM_DST_SIZE is chosen so that they match (square pixels).
        self.ppm_x = self.dst_w / self.real_w
        self.ppm_y = self.dst_h / self.real_h
        self.square_pixels = abs(self.ppm_x - self.ppm_y) < 1e-6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_distance(self, x1, y1, x2, y2):
        """
        Ground distance + lateral offset for a bounding box.

        Uses the bottom-centre of the box as the road contact point, which is
        where a detected vehicle's tyres meet the road.

        Returns (z_ahead_m, x_lateral_m), or (None, None) when the contact
        point falls outside the calibrated road patch.
        """
        u = (x1 + x2) / 2.0
        v = float(y2)
        return self.project_to_ground(u, v)

    def project_to_ground(self, u, v):
        """
        Map a single source-frame pixel onto the ground plane.

        This is the hot path: one 1x1x2 transform per object per frame. It
        does NOT warp the frame.
        """
        point = np.array([[[float(u), float(v)]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(point, self.matrix)[0][0]
        return self._warped_to_metres(float(warped[0]), float(warped[1]))

    def project_points_ground(self, points):
        """
        Batch version of project_to_ground: one cv2.perspectiveTransform call
        for N points instead of N calls. Used by the bird's-eye preview.

        points: iterable of (u, v) source-frame pixels.
        Returns: list of (z_ahead_m, x_lateral_m) | (None, None) per input.
        """
        pts = list(points)
        if not pts:
            return []
        arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(arr, self.matrix).reshape(-1, 2)
        return [self._warped_to_metres(float(p[0]), float(p[1])) for p in warped]

    def warp_birdseye(self, frame):
        """
        Full-frame bird's-eye warp. ONLY call this when the IPM preview panel
        is switched on — it resamples the entire frame and is the single most
        expensive thing in this module on a Raspberry Pi 3B.
        """
        return cv2.warpPerspective(frame, self.matrix, (self.dst_w, self.dst_h))

    def metres_to_birdseye(self, x_m, z_m):
        """
        Ground coordinate in metres -> bird's-eye pixel (px, py).

        The inverse of _warped_to_metres, used to draw object dots on the
        bird's-eye preview. Returns (None, None) if the point falls outside
        the calibrated patch.
        """
        if x_m is None or z_m is None:
            return None, None
        x_m = float(x_m)
        z_m = float(z_m)
        if z_m < 0.0 or z_m > self.real_h:
            return None, None
        if abs(x_m) > (self.real_w / 2.0):
            return None, None

        px = (x_m + self.real_w / 2.0) * self.ppm_x
        py = (self.real_h - z_m) * self.ppm_y
        if not (0.0 <= px <= self.dst_w and 0.0 <= py <= self.dst_h):
            return None, None
        return px, py

    def project_to_image(self, x_m, z_m):
        """
        Inverse: map a ground coordinate in metres back to a source pixel.

        Returns (u, v) as ints, or (None, None) if the point is not inside the
        calibrated patch.
        """
        birdseye = self.metres_to_birdseye(x_m, z_m)
        if birdseye is None:
            return None, None
        px, py = birdseye

        point = np.array([[[px, py]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(point, self.inverse_matrix)[0][0]
        return int(round(float(warped[0]))), int(round(float(warped[1])))

    def get_horizon_y(self):
        """
        Approximate horizon row in the SOURCE frame.

        The far edge and the near edge of the source quad are both parallel to
        the road's left/right direction, so their intersection is the vanishing
        point of the road direction, which lies on the horizon.

        LIMITATION: for a symmetric road those two edges are usually parallel
        in the image, the vanishing point is at infinity, and there is no
        intersection. In that case this falls back to the row of the far edge,
        which is a LOWER bound on the true horizon (the real horizon is above
        the far edge of a finite patch). It is only used by the tests — FCW
        applies its own HORIZON_FRACTION constant to the frame height rather
        than calling this.
        """
        p_far_l, p_far_r, p_near_l, p_near_r = self.src_points
        vx, vy = _line_intersection(p_far_l, p_far_r, p_near_l, p_near_r)
        if vx is None:
            return int(round((p_far_l[1] + p_far_r[1]) / 2.0))
        return int(round(vy))

    def describe(self):
        """One-line human-readable summary of the active calibration."""
        return ("IPM: {0:.1f} m wide x {1:.1f} m deep, {2} px/m "
                "(square={3}), source quad {4}".format(
                    self.real_w, self.real_h, round(self.ppm_x, 2),
                    self.square_pixels, self.src_points))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _warped_to_metres(self, px, py):
        """Bird's-eye pixels -> metres, rejecting points off the patch."""
        if not (np.isfinite(px) and np.isfinite(py)):
            return None, None
        # Inclusive upper bound: a contact point sitting exactly on the far
        # edge lands on px == dst_w / py == dst_h and is still a valid point.
        if not (0.0 <= px <= self.dst_w and 0.0 <= py <= self.dst_h):
            return None, None  # outside the calibrated road region

        z_ahead = self.real_h - (py / self.ppm_y)
        x_lateral = (px / self.ppm_x) - (self.real_w / 2.0)

        if z_ahead < MIN_DISTANCE_M or abs(x_lateral) > MAX_LATERAL_M:
            return None, None

        return round(z_ahead, 2), round(x_lateral, 2)


def _invert_homography(matrix):
    """
    Invert a 3x3 homography.

    NOTE: cv2.invertPerspectiveTransform exists in the C++ API but is NOT
    exposed in the cv2 Python namespace (confirmed against OpenCV 4.11's
    cv2/__init__.pyi), so numpy does the work here. cv2.perspectiveTransform
    accepts any invertible 3x3 matrix; the [2, 2] normalisation is only for
    readability, so the result matches what OpenCV would have returned.

    Raises a clear error for a degenerate quad instead of returning inf/NaN,
    which happens when the four calibration points are collinear or nearly so.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError("homography must be 3x3, got {0}".format(m.shape))

    det = float(np.linalg.det(m))
    if not np.isfinite(det) or abs(det) < 1e-12:
        raise ValueError(
            "IPM calibration is degenerate: the 4 source points are collinear "
            "or too close together, so the homography cannot be inverted. "
            "Re-pick IPM_SRC_POINTS so they form a clear quadrilateral."
        )

    inv = np.linalg.inv(m)
    if inv[2, 2] != 0 and np.isfinite(inv[2, 2]):
        inv = inv / inv[2, 2]
    return inv.astype(np.float32)


def _line_intersection(p1, p2, p3, p4):
    """
    Intersection of the infinite lines p1->p2 and p3->p4.
    Returns (x, y) or (None, None) when parallel or degenerate.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None, None

    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return (a * (x3 - x4) - (x1 - x2) * b) / denom, \
           (a * (y3 - y4) - (y1 - y2) * b) / denom


# ---------------------------------------------------------------------------
# PREVIOUS METHOD — kept as a commented reference, do not delete blindly.
#
# This is the pinhole ground-plane estimator that shipped before the measured
# homography. It reported metres from *assumed* camera geometry:
#
#     alpha      = atan((v - cy) / fy)
#     theta      = pitch + alpha
#     z_ahead_m  = H_cam / tan(theta)
#     x_lateral  = z_ahead_m * (u - cx) / fx
#
# with H_cam = cam_height_m, pitch = pitch_angle_deg, and fy derived from an
# assumed fov_v_deg. The problem was that none of those were measured, so the
# output looked metric but drifted badly whenever the camera was not mounted
# at the assumed height/pitch or the lens was not the assumed field of view.
# That is exactly what the 4-point homography above fixes.
#
# To A/B compare, re-enable the class below and pass it as the estimator.
# It is NOT used by the pipeline.
#
# class IPMPinholeEstimator:                       # previous method
#     def __init__(self, frame_width=640, frame_height=480,
#                  cam_height_m=1.25, pitch_angle_deg=5.0, fov_v_deg=45.0):
#         self.w = float(frame_width)
#         self.h = float(frame_height)
#         self.cam_height = float(cam_height_m)
#         self.pitch_deg = float(pitch_angle_deg)
#         self.pitch_rad = math.radians(self.pitch_deg)
#         self.cx = self.w / 2.0
#         self.cy = self.h / 2.0
#         fov_v_rad = math.radians(fov_v_deg)
#         self.fy = (self.h / 2.0) / math.tan(fov_v_rad / 2.0)
#         self.fx = self.fy
#
#     def estimate_distance(self, x1, y1, x2, y2):
#         return self.project_to_ground((x1 + x2) / 2.0, float(y2))
#
#     def project_to_ground(self, u, v):
#         alpha = math.atan((v - self.cy) / self.fy)
#         total_angle = self.pitch_rad + alpha
#         if total_angle <= 0.002:
#             return None, None
#         z_ahead = self.cam_height / math.tan(total_angle)
#         if z_ahead > 120.0 or z_ahead < 0.2:
#             return None, None
#         x_lateral = z_ahead * ((u - self.cx) / self.fx)
#         return round(z_ahead, 2), round(x_lateral, 2)
#
# (It also had get_horizon_y() returning cy + fy * tan(-pitch_rad).)
#
# Note: the even older `distance = K / height` pixel-height approximation from
# PLAN.md is gone entirely and never shipped in the pipeline; nothing to keep.
# ---------------------------------------------------------------------------
