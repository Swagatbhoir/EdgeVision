"""
Live integration smoke test — runs the REAL hud_renderer + web_server with a
real OpenCV backend, in a child process, and verifies the cockpit is serving
frames, telemetry, and a working dashboard.

No camera or YOLO model is involved: synthetic frames are pushed through the
real HUD renderer and served over the real Flask app.
"""

import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, r"D:\project\ADASS")

import cv2
import numpy as np

from hud_renderer import HUDRenderer
from web_server import TelemetryHub, create_cockpit_app, default_telemetry

PORT = 5099
W, H = 640, 480
SAMPLES = 60


def check(name, cond, extra=""):
    if not cond:
        raise AssertionError("FAILED: {0} {1}".format(name, extra))
    print("  PASS  {0} {1}".format(name, extra))


def get(path, timeout=10, nbytes=None):
    with urllib.request.urlopen("http://127.0.0.1:{0}{1}".format(PORT, path), timeout=timeout) as r:
        # MJPEG / SSE are infinite streams: never read to EOF.
        return r.status, (r.read(nbytes) if nbytes else r.read()), dict(r.headers)


def post(path, payload, timeout=10):
    req = urllib.request.Request(
        "http://127.0.0.1:{0}{1}".format(PORT, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


print("OpenCV", cv2.__version__)

hub = TelemetryHub()
app = create_cockpit_app(hub)

# Real HUD render, real JPEG encode
renderer = HUDRenderer(W, H)


def make_scene():
    """Synthetic road scene: gradient sky, road, lane lines, lead vehicle."""
    frame = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        frame[y, :] = (180 - y // 4, 150 - y // 5, 120 - y // 6)
    cv2.rectangle(frame, (0, 300), (W, H), (60, 60, 60), -1)
    cv2.line(frame, (200, 300), (110, H), (230, 230, 230), 3)
    cv2.line(frame, (440, 300), (530, H), (230, 230, 230), 3)
    cv2.rectangle(frame, (280, 250, 360, 400), (40, 40, 160), -1)  # lead car
    return frame


SCENARIOS = [
    ("normal", None, (280, 250, 360, 400), True, "NORMAL"),
    ("fcw-warning", {
        "system": "FCW", "level": "WARNING",
        "message": "FORWARD COLLISION WARNING", "subtext": "SLOW DOWN",
        "color_bgr": [22, 115, 249], "sound_pattern": "WARNING",
    }, (280, 250, 360, 400), True, "FCW"),
    ("fcw-danger", {
        "system": "FCW", "level": "DANGER",
        "message": "FORWARD COLLISION WARNING", "subtext": "BRAKE NOW",
        "color_bgr": [68, 68, 239], "sound_pattern": "CRITICAL",
    }, (280, 250, 360, 400), True, "FCW"),
    ("ldw-left", {
        "system": "LDW", "level": "WARNING",
        "message": "LANE DEPARTURE WARNING", "subtext": "DRIFTING LEFT",
        "color_bgr": [153, 72, 236], "sound_pattern": "ADVISORY",
    }, (60, 280, 150, 400), False, "LDW"),
    ("fcdw-depart", {
        "system": "FCDW", "level": "INFO",
        "message": "FRONT VEHICLE MOVING", "subtext": "PROCEED WITH CAUTION",
        "color_bgr": [212, 182, 6], "sound_pattern": "CHIME",
    }, (280, 250, 360, 400), True, "FCDW"),
]


def publisher():
    """Drive the real HUD + hub through every alert scenario."""
    timings = {}

    # Warm up: first ever OpenCV draw call pays font/lazy-init costs and is
    # not representative of steady-state per-frame rendering.
    for _ in range(5):
        renderer.render(make_scene(), tracked_objects=[], lane_lines=None,
                        ldw_direction="NORMAL", active_alert=None,
                        fps=27.4, inference_ms=17.9, now=time.time())

    for name, alert, box, in_path, _ in SCENARIOS:
        frame = make_scene()

        risk = {"fcw-danger": "DANGER", "fcw-warning": "WARNING"}.get(name, "CAUTION")
        objects = [{
            "track_id": 1, "name": "car", "confidence": 0.93, "box": box,
            "risk": risk,
            "distance_m": 6.5 if name == "fcw-danger" else 14.2,
            "lateral_offset_m": 0.1, "ttc": 1.2 if name == "fcw-danger" else 2.8,
            "in_ahead": in_path, "in_path": in_path,
        }]

        def draw(target):
            renderer.render(
                target,
                tracked_objects=objects,
                lane_lines=((110, 470, 200, 300), (530, 470, 440, 300)),
                ldw_direction="DRIFTING LEFT" if name == "ldw-left" else "NORMAL",
                active_alert=alert,
                fps=27.4, inference_ms=17.9, lead_vehicle_id=1,
                camera_source="CAM 0", now=time.time(),
            )

        # Measure the HUD render alone (not scene synthesis), warm. Each
        # sample gets a pristine frame so repeated blending never accumulates.
        base = frame.copy()
        samples = []
        for _ in range(SAMPLES):
            target = base.copy()
            t0 = time.perf_counter()
            draw(target)
            samples.append((time.perf_counter() - t0) * 1000.0)

        timings[name] = {
            "median": float(np.median(samples)),
            "p90": float(np.percentile(samples, 90)),
            "min": float(np.min(samples)),
        }
        hud_ms = timings[name]["median"]

        tel = default_telemetry()
        tel.update({
            "timestamp": time.time(), "fps": 27.4, "inference_ms": 17.9,
            "pipeline_ms": hud_ms + 17.9, "hud_render_ms": hud_ms,
            "frame_size": [W, H], "track_count": len(objects),
            "camera_source": "CAM 0", "active_alert": alert,
            "fcw": {"level": risk if risk in ("DANGER", "WARNING") else "SAFE",
                    "distance_m": objects[0]["distance_m"], "lateral_offset_m": 0.1,
                    "ttc": objects[0]["ttc"], "lead_class": "car", "lead_track_id": 1, "enabled": True},
            "ldw": {"departure_direction": "DRIFTING LEFT" if name == "ldw-left" else "NORMAL",
                    "offset": -0.75 if name == "ldw-left" else 0.02, "status": "WARNING_LEFT",
                    "lanes_detected": True, "alert_message": "", "enabled": True},
            "fcdw": {"state": "DEPARTING" if name == "fcdw-depart" else "IDLE",
                     "alert_active": name == "fcdw-depart",
                     "alert_message": "FRONT VEHICLE MOVING" if name == "fcdw-depart" else "",
                     "stopped_duration": 12.0, "lead_track_id": 1, "enabled": True},
            "objects": objects,
        })

        hub.publish_frame(frame)
        hub.publish_telemetry(tel)
        time.sleep(0.35)

    return timings


threading.Thread(target=lambda: app.run(host="127.0.0.1", port=PORT, threaded=True,
                                        use_reloader=False, debug=False), daemon=True).start()

results = {}
pub_thread = threading.Thread(target=lambda: results.__setitem__("timings", publisher()), daemon=True)
pub_thread.start()
time.sleep(0.8)

print("\n--- dashboard ---")
status, body, headers = get("/")
check("dashboard 200", status == 200)
html = body.decode("utf-8")
check("cockpit shell served", "EDGEVISION" in html)
check("stylesheet linked", "/static/css/dashboard.css" in html)
print("     dashboard size: {0} bytes".format(len(body)))

status, body, _ = get("/static/css/dashboard.css")
check("stylesheet 200", status == 200, "({0} bytes)".format(len(body)))
status, body, _ = get("/static/js/dashboard.js")
check("script 200", status == 200, "({0} bytes)".format(len(body)))

print("\n--- MJPEG video feed ---")
status, body, headers = get("/video_feed", timeout=15, nbytes=60000)
check("video_feed 200", status == 200)
check("multipart MJPEG", headers.get("Content-Type", "").startswith("multipart/x-mixed-replace"))
check("stream produced bytes", len(body) > 2000, "({0} bytes)".format(len(body)))
check("frames are JPEG", b"\xff\xd8" in body[:64])
check("multipart boundary present", b"--frame" in body)
check("more than one frame arrived", body.count(b"--frame") >= 2,
      "({0} frames)".format(body.count(b"--frame")))

print("\n--- telemetry API ---")
status, body, _ = get("/api/telemetry")
tel = json.loads(body.decode("utf-8"))
check("telemetry 200", status == 200)
check("fps reported", tel["fps"] == 27.4)
check("inference reported", tel["inference_ms"] == 17.9)
check("hud render time reported", tel["hud_render_ms"] is not None)
check("active alert is a known ADAS system",
      (tel["active_alert"] or {}).get("system") in ("FCW", "LDW", "FCDW", None),
      str((tel["active_alert"] or {}).get("system")))
check("tracked objects present", len(tel["objects"]) == 1)

print("\n--- SSE telemetry stream ---")
req = urllib.request.Request("http://127.0.0.1:{0}/api/stream/telemetry".format(PORT))
resp = urllib.request.urlopen(req, timeout=15)
check("SSE content type", resp.headers.get("Content-Type", "").startswith("text/event-stream"))
# The publisher thread may already be idle, so read whatever events are buffered
# and parse the first COMPLETE event block rather than a fixed byte count.
raw = b""
try:
    while b"\n\n" not in raw and len(raw) < 65536:
        part = resp.read(512)
        if not part:
            break
        raw += part
except Exception:
    pass
resp.close()

text = raw.decode("utf-8", "replace")
check("SSE emits data frames", text.startswith("data: "))
first_event = text.split("\n\n")[0].strip()
check("first event is a complete SSE block", first_event.startswith("data: "))
sse_payload = json.loads(first_event[len("data: "):])
check("SSE payload is valid telemetry", "fcw" in sse_payload and "ldw" in sse_payload)
check("SSE payload carries the active alert key", "active_alert" in sse_payload)

print("\n--- settings API ---")
status, body = post("/api/settings", {"fcw_enabled": False, "hud_radar": False})
check("settings POST 200", status == 200)
status, body, _ = get("/api/settings")
cfg = json.loads(body.decode("utf-8"))
check("FCW disable persisted", cfg["settings"]["fcw_enabled"] is False)
check("radar layer disable persisted", cfg["hud_layers"]["radar"] is False)

print("\n--- scenario simulator API ---")
for scenario, expect_sys in [("fcw-danger", "FCW"), ("ldw-left", "LDW"),
                              ("fcdw-depart", "FCDW")]:
    status, body = post("/api/simulate_alert", {"scenario": scenario})
    check("simulate '{0}' 200".format(scenario), status == 200)
    status, body, _ = get("/api/telemetry")
    tel = json.loads(body.decode("utf-8"))
    got = (tel.get("active_alert") or {}).get("system")
    check("simulate '{0}' surfaces {1}".format(scenario, expect_sys), got == expect_sys)
    check("simulate '{0}' is flagged as simulated".format(scenario),
          tel.get("simulated") is True)

# "normal" clears the override and hands control back to the live pipeline,
# which is still showing its own FCDW scenario at this point.
status, body = post("/api/simulate_alert", {"scenario": "normal"})
check("simulate 'normal' 200", status == 200)
status, body, _ = get("/api/telemetry")
tel = json.loads(body.decode("utf-8"))
check("simulate 'normal' clears the simulated flag", tel.get("simulated") is None)
got = (tel.get("active_alert") or {}).get("system")
check("simulate 'normal' hands control back to live telemetry",
      got in ("FCW", "LDW", "FCDW", None), "live alert = {0}".format(got))
check("live telemetry is what the hub actually holds",
      json.dumps(tel, sort_keys=True) == json.dumps(hub.get_telemetry(), sort_keys=True))

print("\n--- alert history ---")
status, body, _ = get("/api/alerts/history")
alerts = json.loads(body.decode("utf-8"))["alerts"]
check("alert history recorded", len(alerts) > 0, "({0} events)".format(len(alerts)))
systems = sorted(set(a["system"] for a in alerts))
check("history spans multiple systems", len(systems) >= 2, str(systems))
messages = set(a["message"] for a in alerts)
check("history captured the collision warning", "FORWARD COLLISION WARNING" in messages, str(sorted(messages)))

print("\n--- HUD render budget (warm median of {0} frames @ 640x480) ---".format(SAMPLES))
pub_thread.join(timeout=60)
timings = results.get("timings", {})
check("publisher completed every scenario", len(timings) == 5,
      "({0}/5)".format(len(timings)))

# Budget from the design plan: < 2.5 ms of HUD rendering per frame so the
# overlay never becomes the pipeline bottleneck next to YOLO inference.
BUDGET_MS = 2.5
worst = 0.0
for name in ("normal", "fcw-warning", "fcw-danger", "ldw-left", "fcdw-depart"):
    if name in timings:
        t = timings[name]
        worst = max(worst, t["median"])
        print("     {0:<14} median {1:5.2f} ms   p90 {2:5.2f} ms   min {3:5.2f} ms".format(
            name, t["median"], t["p90"], t["min"]))
check("worst-case HUD render stays inside the {0}ms budget".format(BUDGET_MS),
      worst < BUDGET_MS, "= {0:.2f}ms".format(worst))

print("\nALL LIVE COCKPIT INTEGRATION CHECKS PASSED")

# The Flask dev-server runs on a daemon thread that is still blocked in
# accept() at this point; letting the interpreter finalize around it makes the
# process exit non-zero on Windows even though every check passed.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
