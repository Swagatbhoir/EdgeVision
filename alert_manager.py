"""
alert_manager.py — Unified Driver Alert Manager for EdgeVision ADAS.

Arbitrates and harmonizes alerts from multiple concurrent ADAS sub-systems:
  1. Forward Collision Warning (FCW)
  2. Front Car Departure Warning (FCDW)
  3. Lane Departure Warning (LDW)

Priority Hierarchy (Industry Standard Automotive ADAS):
  PRIORITY 1: FCW DANGER     ("FORWARD COLLISION WARNING" - Red, Critical)
  PRIORITY 2: FCW WARNING    ("FORWARD COLLISION WARNING" - Orange, High)
  PRIORITY 3: LDW WARNING    ("LANE DEPARTURE WARNING"    - Orange/Red, High)
  PRIORITY 4: FCDW ALERT     ("FRONT VEHICLE MOVING"      - Cyan/Green, Advisory)
  PRIORITY 5: FCW CAUTION    ("CAUTION"                   - Yellow, Advisory)
  PRIORITY 6: NORMAL         ("NORMAL / SAFE"             - Green/Neutral)

Acoustic policy: the driver is alerted AUDIBLY by FCW DANGER ("BRAKE NOW")
only. Every other level is visual-only (sound_pattern "NONE"), so the cabin is
not filled with advisory chimes that mask the one warning that matters. The
non-critical beep branches are still in _trigger_acoustic so the policy is a
one-word change per level if it ever needs revisiting.

Provides smooth UI banner formatting and acoustic alert triggering
without flickering or overlapping audio beeps.
"""

import time

try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False


class AlertManager:
    """Arbitrates multiple ADAS alerts into a single unified driver output."""

    def __init__(self, alert_cooldown=2.0):
        self.cooldown = alert_cooldown
        self.last_sound_time = 0.0

        # Current active display output
        self.active_alert = None  # None or dict: {type, message, level, color, priority}
        self.active_system = "NONE"

    def update(self, fcw_level, fcdw_active, fcdw_message, ldw_active, ldw_message, ldw_dir="NONE", now=None):
        """
        Evaluate all inputs and select the highest priority alert.
        now: float timestamp (default time.time())
        """
        if now is None:
            now = time.time()

        candidates = []

        # 1. FCW Candidates
        if fcw_level == "DANGER":
            candidates.append({
                "system": "FCW",
                "priority": 1,
                "message": "FORWARD COLLISION WARNING",
                "subtext": "BRAKE NOW",
                "level": "DANGER",
                "color_bgr": (0, 0, 255),       # Bright Red
                "sound_pattern": "CRITICAL",
            })
        elif fcw_level == "WARNING":
            candidates.append({
                "system": "FCW",
                "priority": 2,
                "message": "FORWARD COLLISION WARNING",
                "subtext": "SLOW DOWN",
                "level": "WARNING",
                "color_bgr": (0, 140, 255),     # Orange
                "sound_pattern": "NONE",        # visual-only; beeps reserved for BRAKE NOW
            })

        # 2. LDW Candidates
        if ldw_active:
            sub = f"DRIFTING {ldw_dir}" if ldw_dir != "NONE" else "CHECK LANE"
            candidates.append({
                "system": "LDW",
                "priority": 3,
                "message": ldw_message if ldw_message else "LANE DEPARTURE WARNING",
                "subtext": sub,
                "level": "WARNING",
                "color_bgr": (0, 165, 255),     # Amber
                "sound_pattern": "NONE",        # visual-only; beeps reserved for BRAKE NOW
            })

        # 3. FCDW Candidates
        if fcdw_active:
            candidates.append({
                "system": "FCDW",
                "priority": 4,
                "message": fcdw_message if fcdw_message else "FRONT VEHICLE MOVING",
                "subtext": "PROCEED WITH CAUTION",
                "level": "INFO",
                "color_bgr": (255, 200, 0),     # Bright Cyan/Sky Blue
                "sound_pattern": "NONE",        # visual-only; beeps reserved for BRAKE NOW
            })

        # 4. FCW Caution
        if fcw_level == "CAUTION":
            candidates.append({
                "system": "FCW",
                "priority": 5,
                "message": "CAUTION",
                "subtext": "VEHICLE AHEAD",
                "level": "CAUTION",
                "color_bgr": (0, 230, 255),     # Yellow
                "sound_pattern": "NONE",
            })

        # Select highest priority (lowest numerical priority value)
        if candidates:
            best = min(candidates, key=lambda c: c["priority"])
            self.active_alert = best
            self.active_system = best["system"]
        else:
            self.active_alert = None
            self.active_system = "NORMAL"

        # Trigger acoustic alert if cooldown expired
        self._trigger_acoustic(now)

        return self.active_alert

    def _trigger_acoustic(self, now):
        """
        Plays a non-blocking sound based on the active alert's pattern.

        Only FCW DANGER ("BRAKE NOW") currently uses a non-NONE pattern, so in
        practice this fires once per cooldown window during a hard closing
        situation. The other branches are retained so the acoustic policy can
        be widened again by changing a single "sound_pattern" string above.

        NOTE: `winsound` is Windows-only, so HAS_WINSOUND is False on Linux
        and this is silent on a Raspberry Pi. A GPIO buzzer or `aplay` would
        need to be wired in — not done here because the beep rule is
        deliberately untouched.
        """
        if not HAS_WINSOUND or self.active_alert is None:
            return

        pattern = self.active_alert.get("sound_pattern", "NONE")
        if pattern == "NONE":
            return

        if now - self.last_sound_time < self.cooldown:
            return

        try:
            if pattern == "CRITICAL":
                # Rapid high-frequency emergency beep
                winsound.Beep(1200, 250)
                winsound.Beep(1600, 250)
                self.last_sound_time = now
            elif pattern == "WARNING":
                # Single sharp alert beep
                winsound.Beep(1000, 300)
                self.last_sound_time = now
            elif pattern == "ADVISORY":
                # Double lower-frequency pulse for lane departure
                winsound.Beep(800, 200)
                self.last_sound_time = now
            elif pattern == "CHIME":
                # Friendly chime for front car departure
                winsound.Beep(600, 200)
                winsound.Beep(900, 300)
                self.last_sound_time = now
        except Exception:
            pass
