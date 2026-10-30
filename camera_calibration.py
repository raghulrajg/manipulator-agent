"""
camera_calibration.py
----------------------
Measure where the wrist camera really sits on the flange, from points the
camera saw and the tool tip then touched, and save it as
camera_calibration.json, which manipulator_kinematics then uses instead of
the camera table (for the twin, the pick, everything).

Each pair is one object:
  camera   its position from FoundationPose, in the camera frame [m]
  tip      the tool tip put on that object's centre, base frame (FK)
  where the camera was when it saw the object, one of:
    joints       the arm's joints then [servo deg], or
    object_base  the twin's "object at" for those camera numbers [mm]
                 (the twin placed it with the camera table; pairs given
                 this way must all come from the same arm pose)

At least 3 pairs, better 5 or more, spread over the camera's view (not in
a line). The fit is a rigid transform; the result says how far each pair
is off and, with 4 or more, how far each one is off when left out of the
fit (what to expect on a new object).

The digital twin records pairs for you (Camera view, Calibration). By hand:
    python3 camera_calibration.py pairs.json            # solve and show
    python3 camera_calibration.py pairs.json --save     # ... and use it
    python3 camera_calibration.py --show                # what is in use
    python3 camera_calibration.py --remove              # back to the table

pairs.json:
    {"pairs": [
      {"camera": [-0.168, -0.012, 0.337], "object_base_mm": [-225, 105.5, 41.4],
       "tip_mm": [-391.2, -62.8, 38.3]},
      {"camera": [0.016, -0.007, 0.288], "joints": [-180, 0, -90, 0, 90, 90],
       "tip_mm": [-399.6, 123.1, 89.1]}, ...]}
"""

import argparse
import json
import math
import os
import time

import numpy as np

import manipulator_kinematics as mk


class CalibrationError(ValueError):
    pass


def _rigid_fit(P, Q):
    """R, t minimising |R p + t - q| over the pairs (Kabsch)."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def _vec(pair, key, n, scale=1.0):
    v = pair.get(key)
    if v is None:
        return None
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.shape != (n,) or not np.all(np.isfinite(v)):
        raise CalibrationError(f"{key} must be {n} numbers")
    return v * scale


def flange_poses(pairs):
    """T_base_flange where each pair's camera picture was taken."""
    poses, by_object = [None] * len(pairs), []
    for i, p in enumerate(pairs):
        joints = _vec(p, "joints", 6)
        if joints is not None:
            poses[i] = mk.fk(mk.joint_map.servo_to_visual(joints), "flange")
        elif _vec(p, "object_base_mm", 3) is not None:
            by_object.append(i)
        else:
            raise CalibrationError(f"pair {i + 1}: give joints or object_base_mm (where the camera was)")
    notes = []
    if by_object:
        if len(by_object) < 3:
            raise CalibrationError("pairs given with object_base_mm: at least 3 (they fix the camera pose)")
        cam = np.array([_vec(pairs[i], "camera", 3) for i in by_object])
        obj = np.array([_vec(pairs[i], "object_base_mm", 3, 1e-3) for i in by_object])
        R, t = _rigid_fit(cam, obj)
        off = np.linalg.norm(cam @ R.T + t - obj, axis=1) * 1000
        if off.max() > 5 and len(by_object) >= 4:
            # one pair that disagrees with all the others: fix the camera pose without it
            fits = []
            for k in range(len(by_object)):
                keep = [j for j in range(len(by_object)) if j != k]
                Rk, tk = _rigid_fit(cam[keep], obj[keep])
                fits.append((np.linalg.norm(cam[keep] @ Rk.T + tk - obj[keep], axis=1).max() * 1000, k, Rk, tk))
            worst_rest, k, Rk, tk = min(fits, key=lambda f: f[0])
            if worst_rest < 3:
                R, t = Rk, tk
                off = np.linalg.norm(cam @ R.T + t - obj, axis=1) * 1000
        for i, e in zip(by_object, off):
            if e > 5:
                notes.append(f"pair {i + 1}: the twin's object_base is {e:.0f} mm off the camera pose the "
                             "others give (arm in another pose, or a typing slip?), so it is not used to "
                             "find the camera pose")
        T_base_camera = mk.pose(R, t)
        T_base_flange = T_base_camera @ np.linalg.inv(mk.T_FLANGE_CAMERA_DH)   # the twin used the table
        for i in by_object:
            poses[i] = T_base_flange
    return poses, notes


def solve(pairs):
    """Pairs -> the camera's mounting on the flange and how well it fits."""
    if len(pairs) < 3:
        raise CalibrationError("at least 3 pairs (better 5 or more)")
    poses, notes = flange_poses(pairs)
    P = np.array([_vec(p, "camera", 3) for p in pairs])
    tips = []
    for p in pairs:
        tip = _vec(p, "tip_mm", 3, 1e-3)
        tip = _vec(p, "tip", 3) if tip is None else tip
        if tip is None:
            raise CalibrationError("each pair needs the tool tip: tip_mm (or tip in m)")
        tips.append(tip)
    if np.any(P[:, 2] <= 0) or np.any(np.linalg.norm(P, axis=1) > 2):
        raise CalibrationError("camera positions are in meters, in front of the camera (z > 0)")
    if np.linalg.svd(P - P.mean(0), compute_uv=False)[1] < 0.01:
        raise CalibrationError("the objects lie (nearly) in a line: spread them over the camera's view")
    # each tip in the flange frame at its detection pose: then T_flange_camera maps P onto them
    Q = np.array([(np.linalg.inv(T) @ np.append(tip, 1.0))[:3] for T, tip in zip(poses, tips)])
    R, t = _rigid_fit(P, Q)
    T = mk.pose(R, t)
    residual = np.linalg.norm(P @ R.T + t - Q, axis=1) * 1000
    left_out = []
    if len(pairs) >= 4:
        for i in range(len(pairs)):
            keep = [j for j in range(len(pairs)) if j != i]
            Ri, ti = _rigid_fit(P[keep], Q[keep])
            left_out.append(float(np.linalg.norm(Ri @ P[i] + ti - Q[i]) * 1000))
    # compared with the camera table: the turn about the view axis, the tilt, the shift
    C = np.linalg.inv(mk.T_FLANGE_CAMERA_DH) @ T
    before = np.array([np.linalg.norm(mk.T_FLANGE_CAMERA_DH[:3, :3] @ p + mk.T_FLANGE_CAMERA_DH[:3, 3] - q)
                       for p, q in zip(P, Q)]) * 1000
    return {
        "T_flange_camera": T.round(8).tolist(),
        "pairs": len(pairs),
        "residual_mm": np.round(residual, 1).tolist(),
        "left_out_mm": [round(e, 1) for e in left_out],
        "camera_table_error_mm": np.round(before, 1).tolist(),
        "turn_deg": round(math.degrees(math.atan2(C[1, 0], C[0, 0])), 1),
        "tilt_deg": round(math.degrees(math.acos(float(np.clip(C[2, 2], -1, 1)))), 1),
        "shift_mm": np.round(C[:3, 3] * 1000, 1).tolist(),
        "notes": notes,
    }


def describe(result):
    lines = [f"{result['pairs']} pairs. Against the camera table, the camera is turned "
             f"{result['turn_deg']:+.1f} deg about its view axis, tilted {result['tilt_deg']:.1f} deg, "
             f"shifted {result['shift_mm']} mm (camera frame).",
             f"  error with the camera table [mm]:   {result['camera_table_error_mm']}",
             f"  error with this calibration [mm]:   {result['residual_mm']}"]
    if result["left_out_mm"]:
        lines.append(f"  each pair left out of the fit [mm]: {result['left_out_mm']}  "
                     "(what to expect on a new object)")
    lines += [f"  note: {n}" for n in result["notes"]]
    return "\n".join(lines)


def save(result, pairs=None, path=None):
    """Write camera_calibration.json and use it from now on (this process)."""
    path = path or mk.CAMERA_CALIBRATION_FILE
    data = {"T_flange_camera": result["T_flange_camera"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fit": {k: result[k] for k in ("pairs", "residual_mm", "left_out_mm", "turn_deg", "tilt_deg",
                                           "shift_mm")},
            "measurements": pairs or []}
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    mk.set_camera_calibration(data["T_flange_camera"], dict(data, file=os.path.abspath(path)))
    return path


def main():
    p = argparse.ArgumentParser(description="Calibrate the wrist camera's mounting from camera / tool tip pairs.")
    p.add_argument("pairs", nargs="?", help="JSON file with the pairs (see the top of this file)")
    p.add_argument("--skip", type=int, nargs="+", default=[], metavar="N", help="leave out these pairs (1-based)")
    p.add_argument("--save", action="store_true", help=f"write {os.path.basename(mk.CAMERA_CALIBRATION_FILE)} "
                   "and use it from now on")
    p.add_argument("--show", action="store_true", help="the calibration in use")
    p.add_argument("--remove", action="store_true", help="delete the calibration: back to the camera table")
    args = p.parse_args()
    np.set_printoptions(precision=4, suppress=True)
    if args.remove:
        if os.path.exists(mk.CAMERA_CALIBRATION_FILE):
            os.remove(mk.CAMERA_CALIBRATION_FILE)
            print(f"removed {mk.CAMERA_CALIBRATION_FILE}: the camera table is used")
        else:
            print("no calibration: the camera table is used")
        return
    if args.show or not args.pairs:
        cal = mk.camera_calibration
        if cal is None:
            print("no calibration: the camera table (camera_tranformation.py) is used")
        else:
            print(f"{cal.get('file')} ({cal.get('created', '?')}): fit {cal.get('fit')}")
        print("T_flange_camera in use:\n", mk.T_FLANGE_CAMERA)
        return
    with open(args.pairs) as f:
        data = json.load(f)
    pairs = data["pairs"] if isinstance(data, dict) else data
    pairs = [q for i, q in enumerate(pairs, 1) if i not in args.skip]
    try:
        result = solve(pairs)
    except CalibrationError as e:
        raise SystemExit(f"cannot calibrate: {e}")
    print(describe(result))
    if args.save:
        print(f"saved {save(result, pairs)}: the twin and the pick use it from their next start")
    else:
        print("(not saved: add --save to use it)")


if __name__ == "__main__":
    main()
