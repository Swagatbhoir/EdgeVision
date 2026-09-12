"""
fcdw.py — Front Car Departure Warning (FCDW) module for EdgeVision ADAS.

Pure detection -> departure logic. This module has NO camera or YOLO dependency:
main.py feeds it the YOLO track results (boxes + classes) and it returns whether
a front vehicle departure alert should be presented to the driver.

Concept (see ReadME.md Phase 4):
  1. Lead vehicle directly ahead is stationary (e.g. at a red light or queue).
  2. The system confirms the lead vehicle has been stopped for a minimum duration.
  3. The lead vehicle begins moving forward/away (bounding box moves upward toward
     horizon and/or shrinks in height as distance increases).
  4. Once movement is confirmed over multiple consecutive frames, trigger:
     "FRONT VEHICLE MOVING"
  5. The alert persists for a brief configurable duration (e.g. 3-4 seconds)
     and then resets to avoid nuisance alarms.

Resilience:
  - Temporal smoothing & confirmation frames prevent false triggers from
    single-frame bounding box jitter.
  - Track loss grace period prevents a single missed detection from discarding
    the departure state.
"""

# ---------------------------------------------------------------------------
# Configuration (tunable and explainable for academic viva)
# ---------------------------------------------------------------------------

# Vehicle classes that can qualify as a lead vehicle
LEAD_VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

# Geometry: lead vehicle must be within central corridor and road area
CORRIDOR_HALF_WIDTH_FRAC = 0.30   # +/- 30% of frame width from center
MIN_LEAD_HEIGHT_FRAC = 0.10       # must be reasonably close to be our queue lead
MAX_LEAD_HEIGHT_FRAC = 0.85

# Minimum time (seconds) lead vehicle must remain stationary before departure
# tracking arms (e.g. queue or traffic light stop)
MIN_STOPPED_DURATION = 1.5

# Movement thresholds (in normalized frame coordinates)
# Center Y moving upward (negative dy in image coords) = moving away into distance
# Height shrinking (negative dh) = distance increasing
STATIONARY_Y_VELOCITY_THRESH = 0.015   # normalized px/s - below this is stationary
DEPARTURE_Y_DISPLACEMENT_FRAC = 0.035  # upward move of at least 3.5% frame height
DEPARTURE_HEIGHT_SHRINK_FRAC = 0.10    # or box height shrunk by at least 10%
DEPARTURE_MIN_FRAMES = 4               # consecutive confirming frames

# Track loss grace period (seconds) before giving up on a departing vehicle
TRACK_GRACE_PERIOD = 0.8

# Alert display duration (seconds)
ALERT_DURATION = 3.5

# Cooldown after an alert before a new departure can trigger (seconds)
ALERT_COOLDOWN = 6.0


class FrontCarDepartureWarning:
    """Tracks the lead vehicle ahead when stopped and triggers departure alerts."""

    def __init__(self, frame_width, frame_height):
        self.frame_w = frame_width
        self.frame_h = frame_height

        # State machine: "IDLE", "TRACKING_STOPPED", "DEPARTING", "ALERT"
        self.state = "IDLE"
        self.alert_active = False
        self.alert_message = ""

        # Lead vehicle tracking
        self.lead_track_id = None
        self.stopped_start_time = None
        self.baseline_y2 = None
        self.baseline_height = None
        self.last_seen_time = None

        # Departure confirmation counter
        self.departing_frames = 0

        # Timing
        self.alert_start_time = 0.0
        self.last_alert_finish_time = -100.0

        # Primary info for visualization
        self.lead_info = None

    def update(self, detections, now, ego_stopped=True):
        """
        detections: list of dicts with keys track_id, class_name, x1, y1, x2, y2, confidence
        now: float monotonic timestamp
        ego_stopped: bool, True if host vehicle is assumed/known stopped (default True for dashcam)
        """
        # 1. Manage active alert timeout
        if self.alert_active:
            if now - self.alert_start_time > ALERT_DURATION:
                self.alert_active = False
                self.alert_message = ""
                self.state = "IDLE"
                self.last_alert_finish_time = now
                self._reset_lead()
            else:
                return self

        # 2. If in cooldown after recent alert, remain IDLE
        if now - self.last_alert_finish_time < ALERT_COOLDOWN:
            self.state = "IDLE"
            self._reset_lead()
            return self

        # 3. Find candidate lead vehicle directly ahead
        lead_cand = self._find_lead_vehicle(detections)
        self.lead_info = lead_cand

        if lead_cand is None:
            # Handle grace period if currently tracking
            if self.state in ("TRACKING_STOPPED", "DEPARTING"):
                if self.last_seen_time and (now - self.last_seen_time > TRACK_GRACE_PERIOD):
                    self.state = "IDLE"
                    self._reset_lead()
            else:
                self.state = "IDLE"
                self._reset_lead()
            return self

        tid = lead_cand["track_id"]
        h = lead_cand["height"]
        cy = lead_cand["cy"]
        y2 = lead_cand["y2"]
        self.last_seen_time = now

        # 4. State Machine logic
        if self.state == "IDLE":
            if ego_stopped:
                # Start tracking this vehicle as stopped candidate
                self.lead_track_id = tid
                self.stopped_start_time = now
                self.baseline_y2 = y2
                self.baseline_height = h
                self.departing_frames = 0
                self.state = "TRACKING_STOPPED"

        elif self.state == "TRACKING_STOPPED":
            if tid != self.lead_track_id:
                # Lead vehicle changed, re-initialize
                self.lead_track_id = tid
                self.stopped_start_time = now
                self.baseline_y2 = y2
                self.baseline_height = h
                self.departing_frames = 0
                return self

            stopped_duration = now - self.stopped_start_time
            if stopped_duration < MIN_STOPPED_DURATION:
                # Update baseline to adapt to minor initial settling
                self.baseline_y2 = 0.8 * self.baseline_y2 + 0.2 * y2
                self.baseline_height = 0.8 * self.baseline_height + 0.2 * h
            else:
                # Minimum stopped duration met. Check if lead vehicle is moving away.
                # Moving away: bottom edge y2 moves up (smaller y2) OR height shrinks
                dy_upward = (self.baseline_y2 - y2) / float(self.frame_h)
                dh_shrink = (self.baseline_height - h) / float(self.baseline_height)

                if dy_upward > DEPARTURE_Y_DISPLACEMENT_FRAC or dh_shrink > DEPARTURE_HEIGHT_SHRINK_FRAC:
                    self.departing_frames += 1
                    if self.departing_frames >= DEPARTURE_MIN_FRAMES:
                        self.state = "DEPARTING"
                else:
                    # Vehicle still stationary, slowly update baseline to accommodate drift
                    self.baseline_y2 = 0.95 * self.baseline_y2 + 0.05 * y2
                    self.baseline_height = 0.95 * self.baseline_height + 0.05 * h
                    self.departing_frames = max(0, self.departing_frames - 1)

        elif self.state == "DEPARTING":
            # Departure confirmed -> Trigger alert
            self.state = "ALERT"
            self.alert_active = True
            self.alert_message = "FRONT VEHICLE MOVING"
            self.alert_start_time = now

        return self

    def _find_lead_vehicle(self, detections):
        """Identifies the closest vehicle in our central driving corridor."""
        candidates = []
        center_x = self.frame_w / 2.0
        corridor_w = self.frame_w * CORRIDOR_HALF_WIDTH_FRAC

        for d in detections:
            cname = d.get("class_name", "")
            if cname not in LEAD_VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Must be inside the central driving lane
            if abs(cx - center_x) > corridor_w:
                continue

            h_frac = h / float(self.frame_h)
            if h_frac < MIN_LEAD_HEIGHT_FRAC or h_frac > MAX_LEAD_HEIGHT_FRAC:
                continue

            candidates.append({
                "track_id": d["track_id"],
                "class_name": cname,
                "box": (x1, y1, x2, y2),
                "width": w,
                "height": h,
                "cx": cx,
                "cy": cy,
                "y2": y2,
                "confidence": d.get("confidence", 0.0)
            })

        if not candidates:
            return None

        # The true lead vehicle is the closest one ahead (largest bottom y2 on the road)
        lead = max(candidates, key=lambda c: c["y2"])
        return lead

    def _reset_lead(self):
        self.lead_track_id = None
        self.stopped_start_time = None
        self.baseline_y2 = None
        self.baseline_height = None
        self.last_seen_time = None
        self.departing_frames = 0
