# EdgeVision ADAS — Development Plan & Current Status

This document is the working assessment + step-by-step plan for the EdgeVision project,
as required by `ReadME.md` (§17 Step 5–7). Update it after every completed phase.

---

## 1. What is already working (verified by inspection)

### Detection pipeline (`main.py`, laptop camera)
```
Camera (index 0)
   ↓
YOLO26 nano model (yolo26n.pt) via ultralytics model.track(persist=True)
   ↓
Bounding boxes + track IDs + class names + confidence
   ↓
Per-object: movement direction, "approaching" (height-growth), relative distance (K/height), TTC
   ↓
Risk label (SAFE/WARNING/DANGER) drawn per object
   ↓
winsound beep on DANGER (2 s cooldown) + big "!!! COLLISION WARNING !!!" overlay
   ↓
Local cv2 window + optional Phone-2 stream (Flask /video_feed on :5000)
```

- YOLO detection + tracking loop works as written.
- Phone-2 web-stream dashboard (Flask, `output_frame` + lock) is present and optional.
- Distances are **relative** (`K / height`, `K = 1000`, uncalibrated) but the UI currently
  prints them as meters — false precision that must be fixed.

### Files
| File | Role |
|---|---|
| `main.py` | Current app: camera → YOLO → alerts → display + phone stream |
| `main_backup.py` | Older saved copy (no stream), kept as backup |
| `yolo26n.pt` | YOLO26 nano weights |
| `.venv` | Python 3.11.9 + ultralytics 8.4.139 + torch 2.14.0 + opencv-python + flask |
| `ReadME.md` | Project instructions (read first) |

---

## 2. What is missing / weak (assessment)

### A. Forward Collision Warning — exists but weak
1. **No path-relevance filter** — any object anywhere in the frame (parked cars, roadside
   objects) can trigger risk logic.
2. **False precision** — `K/height` is printed as "m" without calibration.
3. **No 4-level scale** — README asks SAFE → CAUTION → WARNING → DANGER; code only ever
   shows DANGER prominently (`WARNING` appears only as a small per-box label, never as a
   real warning state).
4. **No temporal smoothing / debouncing** — one noisy frame can flip the warning instantly.
5. **Duplicated risk logic** — the same TTC computation appears twice (per-object and the
   big-screen section), which can disagree.

### B. Front Car Departure Warning — completely missing
No code exists for detecting "our car stopped → vehicle ahead leaves → driver hasn't moved".

### C. Lane Departure Warning — completely missing
No lane-marking detection at all.

### D. Other gaps
- No FPS / inference-time / latency monitoring on screen (README §7).
- Model path & settings are not centralized (README §7).
- Camera source is hardcoded `0` (README §8 wants laptop / USB / phone / Pi camera
  selectable without rewriting ADAS logic).

---

## 3. Development order (from ReadME.md "FINAL PRIORITY")

```text
1. [x] Preserve existing working YOLO detection        (DONE)
2. [x] Make FCW robust                                (DONE)
3. [x] Implement Front Car Departure Warning          (DONE)
4. [x] Implement Lane Departure Warning               (DONE)
5. [x] Test all three together (Unified Alert Manager)(DONE)
6. [ ] Optimize for Raspberry Pi 5 (Next stage)
7. [ ] Additional/novel features (Next stage)
```

---

## 4. Phase plan & Status

### Phase 3 — Robust FCW (COMPLETED)
- [x] New module `fcw.py` — pure risk logic, no camera/YOLO dependency
- [x] Path-relevance filter (central corridor + horizon + vehicle/person classes)
- [x] Relative proximity (box height fraction) — no false meters
- [x] Scale-change time-to-collision `tau = height / growth_rate`
- [x] 4 levels SAFE / CAUTION / WARNING / DANGER with hysteresis + debounce
- [x] Single FCW pipeline used by the display (removed duplicated logic)
- [x] Consistent wording: **"FORWARD COLLISION WARNING"** / "CAUTION"
- [x] Offline unit tests (`tests/test_fcw.py`) — 9 tests passing
- [x] Integrated into `main.py`; camera, stream, winsound beep, FPS overlay intact

### Phase 4 — Front Car Departure Warning (COMPLETED)
- [x] New module `fcdw.py` — tracks stationary queue lead vehicle
- [x] Departure confirmation over multi-frame vertical & scale displacement
- [x] Single missed detection grace period (0.8s) prevents premature dropouts
- [x] Consistent wording: **"FRONT VEHICLE MOVING"**
- [x] Offline unit tests (`tests/test_fcdw.py`) — 5 tests passing
- [x] Integrated into `main.py`

### Phase 5 — Lane Departure Warning (COMPLETED)
- [x] New module `ldw.py` — lightweight CV lane detection (Gaussian -> Canny -> ROI -> Hough -> Polyfit)
- [x] Normalized lateral lane center offset with temporal EMA smoothing
- [x] Hysteresis & confirmation frames prevent false alarms on shadows/cracks
- [x] Transparent green lane corridor + boundary overlay drawing
- [x] Consistent wording: **"LANE DEPARTURE WARNING"**
- [x] Offline unit tests (`tests/test_ldw.py`) — 5 tests passing
- [x] Integrated into `main.py`

### Phase 6 — Unified Driver Alert Manager & E2E Integration (COMPLETED)
- [x] New module `alert_manager.py` — industry-standard ADAS priority arbitration:
      P1: FCW DANGER > P2: FCW WARNING > P3: LDW WARNING > P4: FCDW > P5: FCW CAUTION > P6: NORMAL
- [x] Acoustic pattern management (emergency double-beep, single warning beep, lane advisory pulse, departure chime)
- [x] High-contrast UI banner overlay + bottom multi-system status bar
- [x] Unit tests (`tests/test_alert_manager.py`) — 5 tests passing
- [x] E2E driving scenario integration tests (`tests/test_integration_pipeline.py`) — passing
- [x] Master test runner (`tests/run_all_tests.py`) — 5 suites, 25 tests passing (100%)

---

## 5. Honest testing note (ReadME §12)

- Anything tested on the laptop is labelled **tested on laptop**.
- Anything only verified offline (synthetic frames) is labelled **offline unit test**.
- Raspberry Pi 5 behaviour is **expected**, never claimed as tested.