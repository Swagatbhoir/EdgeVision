"""
fcw.py — Forward Collision Warning (FCW) module for EdgeVision ADAS.

Pure detection -> risk logic. This module has NO camera or YOLO dependency:
main.py feeds it the YOLO track results (boxes + classes) and it returns a
stable warning level for the display/alert system.

Design (simple + explainable, see ReadME.md):
  * Relevance filter: the object's road contact point must be below the
    horizon and inside the central driving corridor, and the class must be a
    road obstacle (car, bus, truck, motorcycle, bicycle, person).
  * Closeness is measured by BOX HEIGHT as a fraction of frame height.
    No calibrated meters exist, so we never claim real distances.
  * Time-to-collision uses the classic vision "tau" from scale change:
        tau = height / (height growth rate)
    This is a RELATIVE time-to-contact estimate, not a metric one.
  * Levels SAFE -> CAUTION -> WARNING -> DANGER use enter/exit threshold
    pairs (hysteresis) plus confirmation frames, so a single noisy YOLO
    frame cannot flicker the warning.

Wording (kept consistent across the app, ReadME.md section 6):
    DANGER  -> "FORWARD COLLISION WARNING"   (big alert + beep)
    WARNING -> "FORWARD COLLISION WARNING"
    CAUTION -> "CAUTION"
"""

# ---------------------------------------------------------------------------
# Configuration (tunable; default values are reasonable for 640x480-class
# dashcam frames and are expressed as fractions of frame height so they
# scale with the camera resolution).
# ---------------------------------------------------------------------------

# Object classes that matter for a forward collision risk (YOLO COCO names).
RELEVANT_CLASSES = {
    "car", "bus", "truck", "motorcycle", "bicycle", "person",
}

# Geometry: an object is "in our driving path" when its road contact point
# (bottom edge centre of the box) is below the horizon and inside the
# central corridor.
HORIZON_FRACTION = 0.35          # top 35% of the frame = sky / far away
PATH_HALF_WIDTH_FRACTION = 0.35  # +/- 35% of frame width around the centre

# Ignore boxes smaller than this fraction of frame height (noise floor).
MIN_HEIGHT_FRACTION = 0.04

# Growth-rate smoothing: tau = raw_height / smoothed_growth.
# Box sizes jitter frame to frame, so the growth rate is measured over a
# GROWTH_MIN_SPAN window and then exponentially smoothed. A single bad
# frame can therefore only push tau low for ONE frame, never enough to
# confirm a warning level (see tests/test_fcw.py).
GROWTH_MIN_SPAN = 0.50       # seconds between the two height samples
GROWTH_EMA_ALPHA = 0.5       # lower = smoother but slower response
# Ignore growth rates below this (px/s) when computing tau (jitter floor).
GROWTH_MIN_RATE = 2.0
# Do not trust tau values larger than this (meaningless when far away).
TTC_MAX = 20.0

# ---- Level thresholds ----------------------------------------------------
# ENTER conditions (fraction of frame height, or tau seconds).
# EXIT conditions are LOOSER on purpose (hysteresis): once a warning is on,
# the situation must visibly improve before it goes away. This prevents
# flickering around a boundary value.

CAUTION_H_ENTER = 0.18
CAUTION_TTC_ENTER = 10.0
CAUTION_H_EXIT = 0.13
CAUTION_TTC_EXIT = 12.0

WARNING_H_ENTER = 0.30
WARNING_TTC_ENTER = 3.5
WARNING_H_EXIT = 0.22
WARNING_TTC_EXIT = 7.0

DANGER_H_ENTER = 0.45
DANGER_TTC_ENTER = 1.8
DANGER_H_EXIT = 0.34
DANGER_TTC_EXIT = 3.0

# Debounce: consecutive frames a candidate level must hold before applying.
# Hardening into a worse level is faster than releasing.
CONFIRM_WORSE_FRAMES = 3
CONFIRM_BETTER_FRAMES = 8

# How long (s) a vanished track's state is kept before it is forgotten.
TRACK_FORGET_SECONDS = 2.0

LEVEL_ORDER = ["DANGER", "WARNING", "CAUTION", "SAFE"]
LEVEL_INDEX = {level: i for i, level in enumerate(LEVEL_ORDER)}


class ForwardCollisionWarning:
    """Tracks relevant objects and produces one stable FCW level per frame."""

    def __init__(self, frame_width, frame_height):
        self.frame_w = frame_width
        self.frame_h = frame_height

        # Per track-id state: (time, raw height) samples + smoothed growth.
        self._track_state = {}

        # Global (per "driver") state machine.
        self.level = "SAFE"
        self._candidate = "SAFE"
        self._confirm_count = 0

        # Info about the most dangerous object (for display).
        self.primary = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections, now):
        """
        detections: list of dicts with keys
            track_id, class_name, confidence, x1, y1, x2, y2 (ints, pixels)
        now: float — monotonic time of this frame.

        Returns self with:
            .level   -> "SAFE" | "CAUTION" | "WARNING" | "DANGER"
            .primary -> dict for the closest relevant object or None:
                        {track_id, class_name, confidence, box, h_frac, tau}
        """
        relevant = [d for d in detections if self._is_relevant(d)]

        self._prune_tracks(relevant, now)

        risks = [
            r for r in (self._evaluate_object(d, now) for d in relevant)
            if r is not None
        ]

        # The most dangerous object drives the level (lowest rank = worst,
        # ties broken by larger box i.e. closer).
        if risks:
            self.primary = min(
                risks,
                key=lambda r: (r["natural_rank"], -(r["h_frac"])),
            )
        else:
            self.primary = None

        self._advance_state()
        return self

    @property
    def is_active(self):
        return self.level in ("CAUTION", "WARNING", "DANGER")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_relevant(self, det):
        """Class + geometry filter: is this object in our driving path?"""
        if det.get("class_name", "") not in RELEVANT_CLASSES:
            return False

        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        width = x2 - x1
        height = y2 - y1

        if width < 1 or height < 1:
            return False
        if height / self.frame_h < MIN_HEIGHT_FRACTION:
            return False

        # Road contact point = bottom edge centre of the box.
        bottom_x = (x1 + x2) / 2.0
        bottom_y = y2

        if bottom_y < self.frame_h * HORIZON_FRACTION:
            return False  # above the horizon -> not on our road

        centre = self.frame_w / 2.0
        corridor = self.frame_w * PATH_HALF_WIDTH_FRACTION
        if abs(bottom_x - centre) > corridor:
            return False  # outside the central driving corridor

        return True

    def _evaluate_object(self, det, now):
        """Raw height + tau for one object. Returns a risk dict or None."""
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        height = y2 - y1
        if height <= 0:
            return None
        track_id = det["track_id"]
        h_frac = height / self.frame_h

        st = self._track_state.setdefault(
            track_id,
            {"samples": [], "growth_smooth": 0.0},
        )

        # Keep a short history of (time, raw height).
        st["samples"].append((now, float(height)))
        while len(st["samples"]) > 60:
            st["samples"].pop(0)

        # tau = height / growth rate. Growth is measured over a window of at
        # least GROWTH_MIN_SPAN seconds (pinhole "time to contact" from scale
        # change) and then exponentially smoothed.
        tau = None
        growth = 0.0
        samples = st["samples"]
        if len(samples) >= 2:
            old = None
            for s in reversed(samples):
                if now - s[0] >= GROWTH_MIN_SPAN:
                    old = s
                    break
            if old is not None:
                dt = now - old[0]
                if dt > 0:
                    raw_growth = (height - old[1]) / dt
                    st["growth_smooth"] = (
                        GROWTH_EMA_ALPHA * raw_growth
                        + (1.0 - GROWTH_EMA_ALPHA) * st["growth_smooth"]
                    )
                    growth = st["growth_smooth"]
                    if growth > GROWTH_MIN_RATE and height > 0:
                        tau = height / growth
                        if tau > TTC_MAX:
                            tau = None

        return {
            "track_id": track_id,
            "class_name": det["class_name"],
            "confidence": det.get("confidence", 0.0),
            "box": (x1, y1, x2, y2),
            "h_frac": h_frac,
            "tau": tau,
            "growth": growth,
            "natural_rank": self._enter_rank(h_frac, tau),
        }

    def _enter_rank(self, h_frac, tau):
        """Level this object would justify right now if the system were calm
        (ENTER conditions only)."""
        if h_frac >= DANGER_H_ENTER or (
            tau is not None and tau < DANGER_TTC_ENTER
        ):
            return LEVEL_INDEX["DANGER"]
        if h_frac >= WARNING_H_ENTER or (
            tau is not None and tau < WARNING_TTC_ENTER
        ):
            return LEVEL_INDEX["WARNING"]
        if h_frac >= CAUTION_H_ENTER or (
            tau is not None and tau < CAUTION_TTC_ENTER
        ):
            return LEVEL_INDEX["CAUTION"]
        return LEVEL_INDEX["SAFE"]

    def _hysteresis_rank(self, current_rank, natural_rank, h_frac, tau):
        """
        Apply the EXIT thresholds when the situation is improving.
        A non-active level with a worsening object applies immediately
        (chip away) — the confirmation counter below does the debouncing.
        When improving, the level may drop only ONE step, and only if the
        current level's EXIT conditions are met (looser than ENTER, so the
        warning does not clear on a borderline value).
        """
        if natural_rank < current_rank:
            return natural_rank  # worsening: use ENTER conditions directly

        if current_rank == LEVEL_INDEX["DANGER"]:
            if h_frac < DANGER_H_EXIT and (tau is None or tau > DANGER_TTC_EXIT):
                return LEVEL_INDEX["WARNING"]
            return LEVEL_INDEX["DANGER"]
        if current_rank == LEVEL_INDEX["WARNING"]:
            if h_frac < WARNING_H_EXIT and (tau is None or tau > WARNING_TTC_EXIT):
                return LEVEL_INDEX["CAUTION"]
            return LEVEL_INDEX["WARNING"]
        if current_rank == LEVEL_INDEX["CAUTION"]:
            if h_frac < CAUTION_H_EXIT and (tau is None or tau > CAUTION_TTC_EXIT):
                return LEVEL_INDEX["SAFE"]
            return LEVEL_INDEX["CAUTION"]
        return LEVEL_INDEX["SAFE"]

    def _advance_state(self):
        """Debounced state machine.

        Worsening candidates need CONFIRM_WORSE_FRAMES consecutive frames,
        improving candidates need CONFIRM_BETTER_FRAMES (slow release).
        """
        if self.primary is None:
            natural_rank = LEVEL_INDEX["SAFE"]
            h_frac = 0.0
            tau = None
        else:
            natural_rank = self.primary["natural_rank"]
            h_frac = self.primary["h_frac"]
            tau = self.primary["tau"]

        current_rank = LEVEL_INDEX[self.level]
        candidate_rank = self._hysteresis_rank(
            current_rank, natural_rank, h_frac, tau
        )
        candidate = LEVEL_ORDER[candidate_rank]

        if candidate_rank == current_rank:
            self._candidate = candidate
            self._confirm_count = 0
            return

        worsening = candidate_rank < current_rank  # lower index = worse
        needed = (
            CONFIRM_WORSE_FRAMES
            if worsening
            else CONFIRM_BETTER_FRAMES
        )

        if candidate != self._candidate:
            self._candidate = candidate
            self._confirm_count = 1
        else:
            self._confirm_count += 1

        if self._confirm_count >= needed:
            self.level = candidate
            self._candidate = candidate
            self._confirm_count = 0

    def _prune_tracks(self, relevant, now):
        """Forget track state for objects not seen for TRACK_FORGET_SECONDS."""
        seen = {d["track_id"] for d in relevant}
        cutoff = now - TRACK_FORGET_SECONDS
        for tid in list(self._track_state):
            st = self._track_state[tid]
            last_seen = st["samples"][-1][0] if st["samples"] else 0.0
            if tid not in seen and last_seen < cutoff:
                del self._track_state[tid]