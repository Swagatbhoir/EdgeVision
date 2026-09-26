"""
hud_renderer.py — High-Performance OpenCV Tactical HUD Renderer for EdgeVision ADAS.

Provides automotive-grade AR overlays, tactical corner-bracket bounding boxes,
dynamic multi-tier alert banners, AR lane carpet, top-down IPM BEV mini-radar,
and cockpit telemetry instrumentation.

Performance Target: < 2.5 ms per frame at 640x480 resolution.
"""

import os

import numpy as np
import time
import math

# OpenCV is loaded lazily (it pulls in native DLLs) so this module imports
# cleanly in headless/offline environments. Every draw helper degrades to a
# no-op when it is unavailable, mirroring the convention already used by
# ldw.py. The live camera pipeline always has cv2 present.
#
# Set EDGEVISION_NO_CV2=1 to force the headless path explicitly (headless CI
# or machines where the OpenCV native libraries cannot be loaded at all).
_CV2 = None


def _cv2():
    global _CV2
    if _CV2 is None:
        if os.environ.get("EDGEVISION_NO_CV2", "") == "1":
            _CV2 = False
            return None
        try:
            import cv2 as _mod
            _CV2 = _mod
        except Exception:
            _CV2 = False
    return _CV2 if _CV2 is not False else None


def has_cv2():
    """True when the OpenCV drawing backend is available."""
    return _cv2() is not None


# ===========================================================================
# AUTOMOTIVE SAFETY COLOR PALETTE (BGR Format for OpenCV)
# ===========================================================================
COLOR_DANGER = (68, 68, 239)        # Crimson Red (#EF4444)
COLOR_WARNING = (22, 115, 249)      # Amber Orange (#F97316)
COLOR_CAUTION = (11, 158, 245)      # Gold Yellow (#F59E0B)
COLOR_SAFE = (129, 185, 16)         # Emerald Green (#10B981)
COLOR_LDW = (153, 72, 236)          # Neon Magenta (#EC4899)
COLOR_FCDW = (212, 182, 6)          # Cyber Cyan (#06B6D4)

COLOR_DARK_PANEL = (20, 15, 8)      # Deep OLED Charcoal
COLOR_TEXT_PRIMARY = (250, 250, 248) # Off-White
COLOR_TEXT_MUTED = (184, 163, 148)   # Muted Slate
COLOR_RADAR_BG = (22, 18, 12)       # Semi-dark radar background
COLOR_RADAR_GRID = (85, 70, 50)     # Subtle radar grid lines


# Reusable scratch buffers. The HUD repaints the same handful of panel shapes
# every frame, so allocating a fresh array per overlay costs more than the
# drawing itself on a Raspberry Pi 5. Buffers are keyed by shape and bounded
# so a pathological mix of panel sizes can never grow memory without limit.
_SCRATCH = {}
_SCRATCH_MAX = 48


def _scratch(shape):
    key = (shape[0], shape[1], shape[2] if len(shape) > 2 else 1)
    buf = _SCRATCH.get(key)
    if buf is None:
        if len(_SCRATCH) >= _SCRATCH_MAX:
            _SCRATCH.clear()
        buf = np.empty(shape, dtype=np.uint8)
        _SCRATCH[key] = buf
    return buf


def draw_text_outlined(img, text, pos, font=None, scale=0.5,
                       color=COLOR_TEXT_PRIMARY, thickness=1, outline_color=(0, 0, 0), outline_thickness=3):
    """Draws crisp text with dark outline for 100% legibility over any road scene."""
    cv2 = _cv2()
    if cv2 is None:
        return
    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, text, (x, y), font, scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_panel(img, pt1, pt2, color=COLOR_DARK_PANEL, alpha=0.70, border_color=None, border_thickness=1):
    """Draws a semi-transparent HUD backdrop panel with optional glowing border."""
    cv2 = _cv2()
    if cv2 is None:
        return
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, int(pt1[0])), max(0, int(pt1[1]))
    x2, y2 = min(w_img, int(pt2[0])), min(h_img, int(pt2[1]))
    if x2 <= x1 or y2 <= y1:
        return

    sub_img = img[y1:y2, x1:x2]
    rect = _scratch(sub_img.shape)
    rect[:] = color
    cv2.addWeighted(rect, alpha, sub_img, 1.0 - alpha, 0, sub_img)

    if border_color is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, border_thickness, cv2.LINE_AA)


def draw_tactical_bbox(img, box, class_name, track_id, risk="SAFE",
                       distance=None, ttc=None, lat_offset=None, is_lead=False, confidence=None):
    """
    Renders tactical HUD corner-brackets and compact telemetry badges
    without overlapping text or obstructing the road scene.
    """
    cv2 = _cv2()
    if cv2 is None:
        return
    x1, y1, x2, y2 = map(int, box)
    w = x2 - x1
    h = y2 - y1

    # Map risk level to brand color
    if risk == "DANGER":
        color = COLOR_DANGER
        border_thick = 2
    elif risk == "WARNING":
        color = COLOR_WARNING
        border_thick = 2
    elif risk == "CAUTION":
        color = COLOR_CAUTION
        border_thick = 1
    else:
        color = COLOR_SAFE
        border_thick = 1

    # 1. Subtle bounding box border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # 2. Tactical corner brackets (top-left, top-right, bottom-left, bottom-right)
    bracket_len = max(6, min(int(min(w, h) * 0.22), 24))
    bracket_thick = 2 if risk in ("SAFE", "CAUTION") else 3

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + bracket_len, y1), color, bracket_thick, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + bracket_len), color, bracket_thick, cv2.LINE_AA)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - bracket_len, y1), color, bracket_thick, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + bracket_len), color, bracket_thick, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + bracket_len, y2), color, bracket_thick, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - bracket_len), color, bracket_thick, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - bracket_len, y2), color, bracket_thick, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - bracket_len), color, bracket_thick, cv2.LINE_AA)

    # 3. Top Header Pill: Track ID & Class Name
    conf_str = f" {int(confidence * 100)}%" if confidence is not None else ""
    lead_marker = "[LEAD] " if is_lead else ""
    header_text = f"{lead_marker}#{track_id} {class_name.upper()}{conf_str}"
    
    font_scale = 0.40
    (tw, th), baseline = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    
    pill_x1 = x1
    pill_y1 = max(0, y1 - th - 8)
    pill_x2 = x1 + tw + 10
    pill_y2 = y1

    draw_panel(img, (pill_x1, pill_y1), (pill_x2, pill_y2),
               color=COLOR_DARK_PANEL, alpha=0.85, border_color=color, border_thickness=1)
    cv2.putText(img, header_text, (pill_x1 + 5, pill_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)

    # 4. Bottom Metric Chip: Distance, Lateral Offset, TTC
    metric_parts = []
    if distance is not None:
        metric_parts.append(f"{distance:.1f}m")
    if lat_offset is not None and abs(lat_offset) > 0.05:
        sign = "+" if lat_offset > 0 else ""
        metric_parts.append(f"X:{sign}{lat_offset:.1f}m")
    if ttc is not None and ttc < 15.0:
        metric_parts.append(f"TTC {ttc:.1f}s")
    elif risk in ("WARNING", "DANGER"):
        metric_parts.append(risk)

    if metric_parts:
        chip_text = " • ".join(metric_parts)
        (cw, ch), _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        chip_x1 = x1
        chip_y1 = y2 + 2
        chip_x2 = x1 + cw + 8
        chip_y2 = y2 + ch + 8

        chip_bg = COLOR_DANGER if risk == "DANGER" else (COLOR_WARNING if risk == "WARNING" else COLOR_DARK_PANEL)
        draw_panel(img, (chip_x1, chip_y1), (chip_x2, chip_y2),
                   color=chip_bg, alpha=0.88, border_color=color, border_thickness=1)
        cv2.putText(img, chip_text, (chip_x1 + 4, chip_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)


def draw_cockpit_banner(frame, active_alert, now=None):
    """
    Draws a responsive, high-urgency ADAS warning banner with pulsing glows,
    automotive icons, and peripheral visual alerts.
    """
    if not active_alert or active_alert.get("system") == "NORMAL":
        return

    cv2 = _cv2()
    if cv2 is None:
        return

    if now is None:
        now = time.time()

    h_frame, w_frame = frame.shape[:2]
    level = active_alert.get("level", "WARNING")
    system = active_alert.get("system", "ADAS")
    message = active_alert.get("message", "WARNING")
    subtext = active_alert.get("subtext", "")
    color = active_alert.get("color_bgr", COLOR_WARNING)

    # 1. Peripheral Vignette Pulse for Critical FCW DANGER
    if level == "DANGER":
        pulse = (math.sin(now * 12.0) + 1.0) / 2.0  # 0.0 to 1.0 pulse
        vignette_alpha = 0.15 + 0.15 * pulse
        # Only the four edge bars carry the red tint, so blend just those
        # strips instead of allocating and mixing a full-frame copy.
        border_w = max(6, int(w_frame * 0.015))
        strips = [
            frame[0:border_w, :],                       # top
            frame[h_frame - border_w:h_frame, :],       # bottom
            frame[:, 0:border_w],                       # left
            frame[:, w_frame - border_w:w_frame],       # right
        ]
        for strip in strips:
            tint = _scratch(strip.shape)
            tint[:] = COLOR_DANGER
            cv2.addWeighted(tint, vignette_alpha, strip, 1.0 - vignette_alpha, 0, strip)

    # 2. Banner Geometry
    is_critical = (level == "DANGER")
    banner_w = int(w_frame * (0.68 if is_critical else 0.58))
    banner_h = max(int(h_frame * (0.13 if is_critical else 0.10)), 50)
    banner_x1 = int((w_frame - banner_w) / 2)
    banner_y1 = int(h_frame * 0.04)
    banner_x2 = banner_x1 + banner_w
    banner_y2 = banner_y1 + banner_h

    # 3. Banner Glass Background
    bg_color = (25, 20, 15)
    if is_critical:
        # Pulsing crimson glow
        glow_int = int(40 + 35 * math.sin(now * 10.0))
        bg_color = (glow_int, 20, 160)
    
    draw_panel(frame, (banner_x1, banner_y1), (banner_x2, banner_y2),
               color=bg_color, alpha=0.88, border_color=color, border_thickness=3 if is_critical else 2)

    # 4. Icon Box on Left of Banner
    icon_w = banner_h - 12
    icon_x1 = banner_x1 + 8
    icon_y1 = banner_y1 + 6
    icon_x2 = icon_x1 + icon_w
    icon_y2 = icon_y1 + icon_w

    if is_critical:
        # Exclamation triangle
        pt1 = (int(icon_x1 + icon_w / 2), icon_y1 + 4)
        pt2 = (icon_x1 + 4, icon_y2 - 4)
        pt3 = (icon_x2 - 4, icon_y2 - 4)
        cv2.fillPoly(frame, [np.array([pt1, pt2, pt3], np.int32)], color)
        cv2.putText(frame, "!", (int(icon_x1 + icon_w / 2 - 4), icon_y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    elif system == "LDW":
        # Directional chevrons
        arrow = "<<" if "LEFT" in subtext else (">>" if "RIGHT" in subtext else "< >")
        cv2.rectangle(frame, (icon_x1, icon_y1), (icon_x2, icon_y2), color, 2, cv2.LINE_AA)
        cv2.putText(frame, arrow, (icon_x1 + 4, icon_y1 + int(icon_w * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    else:
        # ADAS Pill badge icon
        cv2.rectangle(frame, (icon_x1, icon_y1), (icon_x2, icon_y2), color, 2, cv2.LINE_AA)
        cv2.putText(frame, system, (icon_x1 + 4, icon_y1 + int(icon_w * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

    # 5. Banner Typography
    text_x = icon_x2 + 12
    title_scale = 0.68 if is_critical else 0.58
    sub_scale = 0.48 if is_critical else 0.42

    # Primary Title
    cv2.putText(frame, message, (text_x, banner_y1 + int(banner_h * 0.48)),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA)
    # Action Subtext
    if subtext:
        cv2.putText(frame, subtext, (text_x, banner_y1 + int(banner_h * 0.84)),
                    cv2.FONT_HERSHEY_SIMPLEX, sub_scale, color, 1, cv2.LINE_AA)


def draw_ar_lane_corridor(frame, left_line, right_line, departure_direction="NONE"):
    """
    Renders an AR road carpet polygon with smooth alpha blend, sharp boundary
    lines, and dynamic color changes on lane drift.
    """
    if left_line is None or right_line is None:
        return

    cv2 = _cv2()
    if cv2 is None:
        return

    lx_bot, ly_bot, lx_top, ly_top = left_line
    rx_bot, ry_bot, rx_top, ry_top = right_line

    is_drifting = (departure_direction in ("LEFT", "RIGHT", "DRIFTING LEFT", "DRIFTING RIGHT"))
    lane_color = COLOR_LDW if is_drifting else COLOR_SAFE

    # 1. AR Road Carpet (Filled Polygon)
    poly_pts = np.array([
        (lx_bot, ly_bot),
        (lx_top, ly_top),
        (rx_top, ry_top),
        (rx_bot, ry_bot)
    ], dtype=np.int32)

    # Blend only the polygon's bounding box instead of copying the whole frame.
    h_f, w_f = frame.shape[:2]
    x0 = max(0, int(poly_pts[:, 0].min()))
    x1 = min(w_f, int(poly_pts[:, 0].max()) + 1)
    y0 = max(0, int(poly_pts[:, 1].min()))
    y1 = min(h_f, int(poly_pts[:, 1].max()) + 1)
    if x1 > x0 and y1 > y0:
        roi = frame[y0:y1, x0:x1]
        scratch = _scratch(roi.shape)
        np.copyto(scratch, roi)
        cv2.fillPoly(scratch, [poly_pts - (x0, y0)], lane_color)
        alpha = 0.28 if is_drifting else 0.16
        cv2.addWeighted(scratch, alpha, roi, 1.0 - alpha, 0, roi)

    # 2. Lane Boundaries
    thick = 3 if is_drifting else 2
    cv2.line(frame, (lx_bot, ly_bot), (lx_top, ly_top), lane_color, thick, cv2.LINE_AA)
    cv2.line(frame, (rx_bot, ry_bot), (rx_top, ry_top), lane_color, thick, cv2.LINE_AA)

    # 3. Dashed Centerline Guide
    cx_bot = int((lx_bot + rx_bot) / 2)
    cy_bot = int((ly_bot + ry_bot) / 2)
    cx_top = int((lx_top + rx_top) / 2)
    cy_top = int((ly_top + ry_top) / 2)

    num_dashes = 6
    for i in range(num_dashes):
        t1 = i / float(num_dashes)
        t2 = (i + 0.5) / float(num_dashes)
        p1 = (int(cx_bot + t1 * (cx_top - cx_bot)), int(cy_bot + t1 * (cy_top - cy_bot)))
        p2 = (int(cx_bot + t2 * (cx_top - cx_bot)), int(cy_bot + t2 * (cy_top - cy_bot)))
        cv2.line(frame, p1, p2, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)


def draw_top_telemetry_bar(frame, fps, inference_ms, camera_source="CAM 0", now_str=None):
    """
    Renders the sleek top instrumentation bar with EdgeVision branding,
    active camera indicator, system clock, and performance badge.
    """
    cv2 = _cv2()
    if cv2 is None:
        return
    h_frame, w_frame = frame.shape[:2]
    bar_h = 32

    # Top dark strip
    draw_panel(frame, (0, 0), (w_frame, bar_h), color=COLOR_DARK_PANEL, alpha=0.85,
               border_color=(60, 50, 40), border_thickness=1)

    # Left: Brand + Camera Source
    cv2.putText(frame, "EDGEVISION ADAS", (12, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA)
    
    # Camera pill badge
    cam_badge = f"[{camera_source}]"
    cv2.putText(frame, cam_badge, (160, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_FCDW, 1, cv2.LINE_AA)

    # Right: FPS & Latency
    if fps is not None:
        fps_color = COLOR_SAFE if fps >= 24 else (COLOR_WARNING if fps >= 15 else COLOR_DANGER)
        fps_text = f"FPS {fps:4.1f}"
        (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, fps_text, (w_frame - 180, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 2, cv2.LINE_AA)

    if inference_ms is not None:
        inf_text = f"{inference_ms:4.1f}ms"
        cv2.putText(frame, inf_text, (w_frame - 90, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)


def draw_bottom_status_bar(frame, fcw_level="SAFE", ldw_state="NORMAL", fcdw_state="IDLE"):
    """
    Renders the bottom cockpit multi-subsystem status bar with distinct pill cards.
    """
    cv2 = _cv2()
    if cv2 is None:
        return
    h_frame, w_frame = frame.shape[:2]
    bar_y1 = h_frame - 34
    bar_y2 = h_frame - 4

    pills = [
        ("FCW", fcw_level, COLOR_DANGER if fcw_level == "DANGER" else (COLOR_WARNING if fcw_level == "WARNING" else (COLOR_CAUTION if fcw_level == "CAUTION" else COLOR_SAFE))),
        ("LDW", ldw_state if ldw_state != "NORMAL" else "TRACKING", COLOR_LDW if ldw_state != "NORMAL" else COLOR_SAFE),
        ("FCDW", fcdw_state, COLOR_FCDW if fcdw_state == "ALERT" else COLOR_SAFE),
    ]

    curr_x = 12
    for sys_name, status_str, color in pills:
        label = f"{sys_name}: {status_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        pill_w = tw + 16

        draw_panel(frame, (curr_x, bar_y1), (curr_x + pill_w, bar_y2),
                   color=COLOR_DARK_PANEL, alpha=0.82, border_color=color, border_thickness=1)
        cv2.putText(frame, label, (curr_x + 8, bar_y2 - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
        curr_x += pill_w + 10


def draw_bev_radar(frame, tracked_objects, max_dist=30.0, max_lat=4.0):
    """
    Renders an IPM Bird's-Eye View (BEV) Mini-Radar in the bottom-right corner,
    mapping detected obstacles into top-down metric ground coordinates (X, Z).
    """
    cv2 = _cv2()
    if cv2 is None:
        return
    h_frame, w_frame = frame.shape[:2]
    radar_size = 125
    margin = 8
    rx1 = w_frame - radar_size - margin
    ry1 = h_frame - radar_size - margin
    rx2 = rx1 + radar_size
    ry2 = ry1 + radar_size

    # Radar dark background panel
    draw_panel(frame, (rx1, ry1), (rx2, ry2), color=COLOR_RADAR_BG, alpha=0.85,
               border_color=COLOR_RADAR_GRID, border_thickness=1)

    # Radar header
    cv2.putText(frame, "BEV RADAR", (rx1 + 6, ry1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Ego car position (bottom-center of radar)
    ego_cx = rx1 + int(radar_size / 2)
    ego_cy = ry2 - 14

    # Ego car symbol (small blue triangle / box)
    cv2.rectangle(frame, (ego_cx - 4, ego_cy - 6), (ego_cx + 4, ego_cy + 2), COLOR_FCDW, -1)

    # Concentric distance rings (10m, 20m, 30m)
    scale_y = (radar_size - 28) / max_dist
    scale_x = (radar_size / 2 - 10) / max_lat

    for dist_m in (10, 20, 30):
        r_px = int(dist_m * scale_y)
        cv2.ellipse(frame, (ego_cx, ego_cy), (int(r_px * 0.9), r_px), 0, 180, 360, COLOR_RADAR_GRID, 1, cv2.LINE_AA)

    # Ego vehicle central path corridor lines
    cv2.line(frame, (ego_cx - 10, ego_cy), (ego_cx - 10, ry1 + 20), (50, 45, 35), 1, cv2.LINE_AA)
    cv2.line(frame, (ego_cx + 10, ego_cy), (ego_cx + 10, ry1 + 20), (50, 45, 35), 1, cv2.LINE_AA)

    # Plot obstacle blips
    for obj in tracked_objects:
        dist = obj.get("distance_m") or obj.get("distance")
        lat = obj.get("lateral_offset_m") or obj.get("lateral_offset")
        risk = obj.get("risk", "SAFE")

        if dist is None or dist <= 0 or dist > max_dist:
            continue
        if lat is None:
            lat = 0.0

        # Map metric (lat, dist) -> (radar_x, radar_y)
        blip_x = int(ego_cx + lat * scale_x)
        blip_y = int(ego_cy - dist * scale_y)

        blip_x = max(rx1 + 4, min(rx2 - 4, blip_x))
        blip_y = max(ry1 + 16, min(ry2 - 4, blip_y))

        blip_color = COLOR_DANGER if risk == "DANGER" else (COLOR_WARNING if risk == "WARNING" else (COLOR_CAUTION if risk == "CAUTION" else COLOR_SAFE))
        cv2.circle(frame, (blip_x, blip_y), 3, blip_color, -1, cv2.LINE_AA)


class HUDRenderer:
    """Master HUD rendering engine for EdgeVision ADAS."""

    def __init__(self, frame_width=640, frame_height=480):
        self.frame_w = frame_width
        self.frame_h = frame_height

    def render(self, frame, tracked_objects, lane_lines=None, ldw_direction="NORMAL",
               active_alert=None, fps=None, inference_ms=None, lead_vehicle_id=None,
               show_bboxes=True, show_lanes=True, show_radar=True, show_telemetry=True,
               camera_source="CAM 0", now=None):
        """
        Renders the complete tactical cockpit HUD onto the provided frame in-place.
        """
        if _cv2() is None:
            return frame

        if now is None:
            now = time.time()

        # 1. AR Lane Guidance Carpet
        if show_lanes and lane_lines is not None:
            left_line, right_line = lane_lines
            draw_ar_lane_corridor(frame, left_line, right_line, departure_direction=ldw_direction)

        # 2. Tactical Bounding Boxes
        if show_bboxes and tracked_objects:
            for obj in tracked_objects:
                box = obj.get("box", (0, 0, 0, 0))
                cname = obj.get("name", "obj")
                tid = obj.get("track_id", 0)
                risk = obj.get("risk", "SAFE")
                dist = obj.get("distance_m", obj.get("distance"))
                ttc = obj.get("ttc")
                lat = obj.get("lateral_offset_m", obj.get("lateral_offset"))
                conf = obj.get("confidence")
                is_lead = (tid == lead_vehicle_id and lead_vehicle_id is not None)

                draw_tactical_bbox(frame, box, cname, tid, risk=risk,
                                   distance=dist, ttc=ttc, lat_offset=lat,
                                   is_lead=is_lead, confidence=conf)

        # 3. Dynamic Urgency Alert Banner
        if active_alert:
            draw_cockpit_banner(frame, active_alert, now=now)

        # 4. Top Telemetry Header
        if show_telemetry:
            draw_top_telemetry_bar(frame, fps, inference_ms, camera_source=camera_source)

        # 5. Bottom Subsystem Matrix
        if show_telemetry:
            fcw_lvl = active_alert.get("level", "SAFE") if (active_alert and active_alert.get("system") == "FCW") else "SAFE"
            fcdw_st = "ALERT" if (active_alert and active_alert.get("system") == "FCDW") else "IDLE"
            draw_bottom_status_bar(frame, fcw_level=fcw_lvl, ldw_state=ldw_direction, fcdw_state=fcdw_st)

        # 6. Top-Down Bird's-Eye View (BEV) Mini-Radar
        if show_radar and tracked_objects:
            draw_bev_radar(frame, tracked_objects)

        return frame
