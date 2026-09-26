"""
Standalone verification of the IPM homography arithmetic in ipm_distance.py.

Reimplements the homography solve in numpy (same math cv2.getPerspectiveTransform
uses) so the axis conventions and the pixels-per-metre conversion can be checked
without importing cv2, which crashes in this sandbox.
"""
import numpy as np

# --- Mirror of the production constants -------------------------------------
SRC = [(224, 236), (416, 236), (512, 430), (128, 430)]   # far-L, far-R, near-R, near-L
REAL_W = 4.0
REAL_H = 10.0
DST_W, DST_H = 400, 1000

DST = [[0, 0], [DST_W, 0], [DST_W, DST_H], [0, DST_H]]
PPM_X = DST_W / REAL_W
PPM_Y = DST_H / REAL_H

fails = []


def check(label, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + label + ("  " + detail if detail else ""))
    if not cond:
        fails.append(label)


def solve_homography(src, dst):
    """
    8-DOF solve with h8 fixed to 1 — the exact formulation
    cv2.getPerspectiveTransform() uses, so this is a faithful mirror.

        x*h0 + y*h1 + h2 = u * (x*h6 + y*h7 + 1)
        x*h3 + y*h4 + h5 = v * (x*h6 + y*h7 + 1)
    """
    rows, rhs = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    h, *_ = np.linalg.lstsq(np.array(rows, dtype=np.float64),
                            np.array(rhs, dtype=np.float64), rcond=None)
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]], dtype=np.float64)


def close(got, expect, tol=1e-6):
    return (got is not None
            and abs(got[0] - expect[0]) < tol
            and abs(got[1] - expect[1]) < tol)


H = solve_homography(SRC, DST)


def warp(H, pts):
    p = np.array([[[float(a), float(b)]] for a, b in pts], dtype=np.float64)
    q = p.reshape(-1, 2)
    ones = np.ones((q.shape[0], 1))
    homo = np.hstack([q, ones]) @ H.T
    return homo[:, :2] / homo[:, 2:3]


def to_metres(px, py):
    if not (0.0 <= px <= DST_W and 0.0 <= py <= DST_H):
        return None, None
    z = REAL_H - (py / PPM_Y)
    x = (px / PPM_X) - (REAL_W / 2.0)
    if z < 0.0 or abs(x) > 50.0:
        return None, None
    return round(z, 2), round(x, 2)


print("\n=== 1. Pixels-per-metre ===")
check("PPM square (400/4.0 == 1000/10.0)", abs(PPM_X - PPM_Y) < 1e-9,
      "= {0} px/m".format(PPM_X))

print("\n=== 2. The 4 source points land on the 4 destination corners ===")
w = warp(H, SRC)
for i, (name, expect) in enumerate(zip(
        ["far-left", "far-right", "near-right", "near-left"], DST)):
    got = w[i]
    ok = abs(got[0] - expect[0]) < 1e-6 and abs(got[1] - expect[1]) < 1e-6
    check(name, ok, "-> ({0:.1f}, {1:.1f}) expect ({2}, {3})".format(
        got[0], got[1], expect[0], expect[1]))

print("\n=== 3. Corner -> metres has the right sign and range ===")
cases = [("far-left",  0, 0,    (10.0, -2.0)),
         ("far-right", 400, 0,   (10.0,  2.0)),
         ("near-left", 0, 1000,  ( 0.0, -2.0)),
         ("near-right", 400, 1000, (0.0, 2.0))]
for name, px, py, expect in cases:
    got = to_metres(px, py)
    ok = close(got, expect)
    check(name, ok, "-> {0} expect {1}".format(got, expect))

print("\n=== 4. Centre of the patch is straight ahead, on the centreline ===")
got = to_metres(DST_W / 2, DST_H / 2)
check("centre is 5.0 m ahead, 0.0 m lateral", close(got, (5.0, 0.0)), "-> {0}".format(got))

print("\n=== 5. Monotonic: lower in the image = closer ===")
# Sample the centre column of the source image, top to bottom.
col = warp(H, [(320, y) for y in range(180, 481, 20)])
zs = [to_metres(p[0], p[1]) for p in col]
valid = [z for z in zs if z is not None and z[0] is not None]
rejected = len(zs) - len(valid)
check("distance decreases monotonically as y increases",
      len(valid) > 3 and all(valid[i][0] > valid[i + 1][0] for i in range(len(valid) - 1)),
      "{0} ... {1} ({2} sampled, {3} off-patch)".format(
          valid[0][0] if valid else "?", valid[-1][0] if valid else "?", len(zs), rejected))

print("\n=== 6. Round trip: source pixel -> metres -> source pixel ===")
# The estimator reports metres rounded to 2 dp, i.e. quantised to 0.01 m steps
# with up to 0.005 m of error. At 100 px/m that is 0.5 px, so the round trip
# cannot be tighter than that. This checks the homography itself is lossless
# beyond the reporting precision.
Hinv = np.linalg.inv(H)
tolerance = 0.005 * PPM_X + 1e-6
worst = 0.0
for u, v in [(200, 300), (300, 380), (400, 420), (250, 260), (450, 350)]:
    px, py = warp(H, [(u, v)])[0]
    z, x = to_metres(px, py)
    if z is None:
        continue
    back_x = (x + REAL_W / 2.0) * PPM_X
    back_y = (REAL_H - z) * PPM_Y
    homo = np.array([[back_x, back_y, 1.0]]) @ Hinv.T
    rx, ry = homo[0][0] / homo[0][2], homo[0][1] / homo[0][2]
    worst = max(worst, abs(rx - u), abs(ry - v))
check("round trip error within 2-dp reporting bound", worst < tolerance,
      "worst = {0:.3f} px, bound = {1:.3f} px".format(worst, tolerance))

print("\n=== 7. Out-of-region is rejected, not crashed ===")
# Well above the far edge (sky) and far off to the left.
sky = warp(H, [(320, 40)])
offroad = warp(H, [(5, 460)])
check("sky point rejected", to_metres(sky[0][0], sky[0][1]) == (None, None),
      "warped to ({0:.0f}, {1:.0f})".format(sky[0][0], sky[0][1]))
check("far-off-road point rejected", to_metres(offroad[0][0], offroad[0][1]) == (None, None),
      "warped to ({0:.0f}, {1:.0f})".format(offroad[0][0], offroad[0][1]))

print("\n=== 8. _invert_homography (the path that just crashed) ===")


def production_invert(matrix):
    """Mirror of ipm_distance._invert_homography, including normalisation."""
    m = np.asarray(matrix, dtype=np.float64)
    det = float(np.linalg.det(m))
    if not np.isfinite(det) or abs(det) < 1e-12:
        raise ValueError("degenerate homography")
    inv = np.linalg.inv(m)
    if inv[2, 2] != 0 and np.isfinite(inv[2, 2]):
        inv = inv / inv[2, 2]
    return inv.astype(np.float32)


det = float(np.linalg.det(H))
check("det is non-zero for a valid quad", abs(det) > 1e-12, "det = {0:.4e}".format(det))

Hinv_prod = production_invert(H)
check("production inverse is a float32 3x3",
      Hinv_prod.shape == (3, 3) and Hinv_prod.dtype == np.float32,
      "shape {0}, dtype {1}".format(Hinv_prod.shape, Hinv_prod.dtype))
check("normalisation puts inv[2,2] at 1.0", abs(float(Hinv_prod[2, 2]) - 1.0) < 1e-6,
      "inv[2,2] = {0:.6f}".format(float(Hinv_prod[2, 2])))

prod = H @ Hinv_prod.astype(np.float64)
prod = prod / prod[2, 2]
identity_err = float(np.abs(prod - np.eye(3)).max())
# Hinv_prod is float32 (what cv2.perspectiveTransform wants), and float32 has
# ~1e-7 relative precision, so 1e-9 is below the representation floor.
check("H @ Hinv == identity (float32 precision)", identity_err < 1e-6,
      "max abs error = {0:.2e}".format(identity_err))

# The normalised inverse must round-trip identically to the raw one, to within
# the same float32 floor.
worst_norm = 0.0
for u, v in [(320, 420), (200, 350), (450, 320)]:
    z, x = to_metres(*warp(H, [(u, v)])[0])
    bx = (x + REAL_W / 2.0) * PPM_X
    by = (REAL_H - z) * PPM_Y
    h1 = np.array([[bx, by, 1.0]]) @ Hinv.T
    h2 = np.array([[bx, by, 1.0]]) @ Hinv_prod.astype(np.float64).T
    worst_norm = max(worst_norm,
                     abs(h1[0][0] / h1[0][2] - h2[0][0] / h2[0][2]),
                     abs(h1[0][1] / h1[0][2] - h2[0][1] / h2[0][2]))
check("normalisation changes nothing", worst_norm < 1e-3,
      "max divergence = {0:.2e} px (float32 floor)".format(worst_norm))

# Guard arithmetic: feed a genuinely singular matrix. (The test stub's lstsq
# solver happily returns a non-singular minimum-norm solution for collinear
# input, so it is not a faithful stand-in for cv2 here — the determinant check
# itself is what we want to exercise.)
for label, bad in (
        ("zero row", [[1, 0, 0], [0, 0, 0], [0, 0, 1]]),
        ("zero det", [[1, 2, 3], [4, 5, 6], [7, 8, 9]])):
    try:
        production_invert(bad)
        check("degenerate matrix ({0}) raises a clear error".format(label), False,
              "no error raised")
    except ValueError as exc:
        check("degenerate matrix ({0}) raises a clear error".format(label), True,
              str(exc)[:44] + "...")

print("\n" + "=" * 46)
print(" FAILURES: {0}".format(len(fails)))
for f in fails:
    print("   - " + f)
print("=" * 46)
