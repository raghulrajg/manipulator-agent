"""
test_ik_random.py
------------------
Random joint angles -> forward kinematics -> inverse kinematics -> joint
angles, compared with the angles we started from.

For every joint state (servo degrees, inside SERVO_LIMITS_DEG):
  1. T = fk(q, "tool")       the tool tip transformation in the base frame
  2. ik_solutions(T, ref)    every joint state within the limits that reaches
                             T, nearest ref first
  3. compare:
     a. with ref = the original angles (how the robot picks a solution, its
        current joint state being the reference): the nearest solution must
        be the original angles
     b. with ref = an unrelated random joint state: the original must still
        be among the solutions -- so the IK can't pass by echoing its
        reference back
     c. FK of every solution must give T back (position and rotation)

Sections:
  random       uniform random joint states
  limits       one joint exactly at its lower or upper limit
  singular     the arm's singular poses: wrist straight (J5 = 0 / 180 deg,
               J4 and J6 parallel), elbow in line (J3 = 0 deg), shoulder (the
               wrist centre on the cylinder of radius d4 around the J1 axis).
               There, and right next to them, the pose no longer pins every
               joint down on its own: they are judged by pose, and the joint
               difference is printed for information. At the exact wrist
               singularity the joints are checked too, because joint 6 then
               comes from the reference.

Run:
    python3 test_ik_random.py                     # 1000 random poses, first 3 shown
    python3 test_ik_random.py --n 10000 --seed 42 --show 5
"""

import argparse
import math
import sys

import numpy as np

import manipulator_kinematics as _mk

_mk.set_camera_calibration(None)   # the tests check the camera table, not a saved calibration

from manipulator_kinematics import (
    ARM_DH, IKError, SERVO_LIMITS_DEG, WRIST_DH, build_dh_params, fk,
    forward_kinematics_6dof, ik_solutions, joint_map,
)

JOINT_TOL_DEG = 1e-6   # recovered joint angle vs original
POS_TOL_M = 1e-10      # FK(IK(T)) vs T: position ...
ROT_TOL_DEG = 1e-8     # ... and rotation (worst seen ~1e-15 m / 2e-13 deg)
# Inside the IK's wrist-singular band (|sin th5| < 1e-8) joint 6 is chosen
# freely (the reference, or spread over its reachable range), which leaves an
# orientation error of at most |sin th5| rad.
BAND_POS_TOL_M = 1e-8
BAND_ROT_TOL_DEG = 1e-6
# Within these of a singularity the joints are ill-conditioned (pose exact,
# joints off by up to ~1e-4 deg): judge by pose there.
NEAR_SINGULAR_SIN = 1e-4    # |sin th5| (wrist) or |sin th3| (elbow)
NEAR_SINGULAR_R_M = 1e-6    # wrist centre within this of radius d4 (shoulder)

A2, A3, D4 = ARM_DH[1][1], ARM_DH[2][1], ARM_DH[3][2]
D5, D6 = WRIST_DH["flange"][0][2], WRIST_DH["flange"][1][2]
OFFSETS = build_dh_params(np.zeros(6))[:, 3]   # DH theta offsets


def rotation_error_deg(R_a, R_b):
    """Angle of the rotation between R_a and R_b (atan2 form: accurate for
    tiny angles, where acos of the trace is not)."""
    R = R_a.T @ R_b
    s = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    c = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.atan2(s, c))


def wrap180(deg):
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def conditioning(q):
    """How close joint state q (visual rad) is to each singularity."""
    theta = build_dh_params(q)[:, 3]
    T_flange = forward_kinematics_6dof(build_dh_params(q))[0]
    p5 = T_flange[:3, 3] - D6 * T_flange[:3, 2]
    return {"sin_th5": abs(math.sin(theta[4])), "sin_th3": abs(math.sin(theta[2])),
            "shoulder_m": math.hypot(p5[0], p5[1]) - D4}


def round_trip(servo_deg, other_ref_deg, ref_deg=None):
    """FK -> IK -> compare for one joint state (servo degrees). ref_deg is the
    IK reference for check a (default: the original angles)."""
    servo_deg = np.asarray(servo_deg, dtype=float)
    ref_deg = servo_deg if ref_deg is None else np.asarray(ref_deg, dtype=float)
    q = joint_map.servo_to_visual(servo_deg)
    T = fk(q, "tool")
    c = conditioning(q)
    r = {"servo_deg": servo_deg, "T": T, "solutions": [], "error": None, **c,
         "in_band": c["sin_th5"] < 1e-8,
         "ill_conditioned": (min(c["sin_th5"], c["sin_th3"]) < NEAR_SINGULAR_SIN
                             or c["shoulder_m"] < NEAR_SINGULAR_R_M)}
    try:
        solutions = ik_solutions(T, ref_deg)           # no table: pure kinematics
        others = ik_solutions(T, other_ref_deg)
    except IKError as e:
        r["error"] = str(e)
        return r

    pos, rot = [], []
    for s in solutions:
        F = fk(joint_map.servo_to_visual(s), "tool")
        pos.append(float(np.linalg.norm(F[:3, 3] - T[:3, 3])))
        rot.append(rotation_error_deg(F[:3, :3], T[:3, :3]))
    r.update(
        solutions=solutions,
        nearest=solutions[0],
        joint_err_deg=float(np.abs(solutions[0] - servo_deg).max()),
        # how close the closest solution found from the unrelated reference is
        other_ref_err_deg=float(min(np.abs(wrap180(s - servo_deg)).max() for s in others)),
        pos_err_m=max(pos),
        rot_err_deg=max(rot),
    )
    return r


def pose_ok(r):
    if r["error"] is not None:
        return False
    pos_tol, rot_tol = (BAND_POS_TOL_M, BAND_ROT_TOL_DEG) if r["in_band"] else (POS_TOL_M, ROT_TOL_DEG)
    return r["pos_err_m"] <= pos_tol and r["rot_err_deg"] <= rot_tol


def joints_ok(r):
    """Checks a and b; skipped next to a singularity (judged by pose)."""
    if r["error"] is not None:
        return False
    if r["ill_conditioned"]:
        return True
    return r["joint_err_deg"] <= JOINT_TOL_DEG and r["other_ref_err_deg"] <= JOINT_TOL_DEG


def describe_failure(r):
    if r["error"]:
        return f"IKError: {r['error']}"
    return (f"joint error {r['joint_err_deg']:.1e} deg, from an unrelated reference "
            f"{r['other_ref_err_deg']:.1e} deg, position {r['pos_err_m']:.1e} m, "
            f"rotation {r['rot_err_deg']:.1e} deg")


def show(index, r):
    angles = {"float_kind": lambda v: f"{v:9.4f}"}
    small = {"float_kind": lambda v: f"{v:9.1e}"}
    print(f"Sample {index}")
    print(f"  random joint angles (servo deg): {np.array2string(r['servo_deg'], formatter=angles)}")
    print("  FK tool tip transformation T_base_tool:")
    for row in r["T"]:
        print("     " + "  ".join(f"{v:9.5f}" for v in row))
    if r["error"]:
        print(f"  IK FAILED: {r['error']}\n")
        return
    print(f"  IK found {len(r['solutions'])} solutions within the joint limits (servo deg):")
    for k, s in enumerate(r["solutions"], start=1):
        mark = "  <- nearest the original" if k == 1 else ""
        print(f"     {k}  {np.array2string(s, formatter=angles)}{mark}")
    print(f"  IK nearest - original (deg):     {np.array2string(r['nearest'] - r['servo_deg'], formatter=small)}")
    print(f"  from an unrelated reference the original is found again to {r['other_ref_err_deg']:.1e} deg")
    print(f"  every solution reproduces T: position error <= {r['pos_err_m']:.1e} m, "
          f"rotation error <= {r['rot_err_deg']:.1e} deg")
    print(f"  {'PASS' if joints_ok(r) and pose_ok(r) else 'FAIL'}\n")


def summarize(title, results):
    """Print the summary of one batch; return the failed results."""
    solved = [r for r in results if r["error"] is None]
    judged = [r for r in solved if not r["ill_conditioned"]]

    def worst(rs, key):
        return max((r[key] for r in rs), default=float("nan"))

    n = len(results)
    failed = [r for r in results if not (joints_ok(r) and pose_ok(r))]
    print(f"==== {title} ====")
    print(f"IK raised IKError                   : {n - len(solved)}/{n}")
    print(f"recovered the original joint angles : {sum(joints_ok(r) for r in solved)}/{len(solved)}   "
          f"(worst {worst(judged, 'joint_err_deg'):.1e} deg; from an unrelated reference "
          f"{worst(judged, 'other_ref_err_deg'):.1e} deg"
          + (f"; {len(solved) - len(judged)} next to a singularity judged by pose" if len(judged) < len(solved) else "")
          + ")")
    print(f"every IK solution reproduces T      : {sum(pose_ok(r) for r in solved)}/{len(solved)}   "
          f"(worst position {worst(solved, 'pos_err_m'):.1e} m, rotation {worst(solved, 'rot_err_deg'):.1e} deg)")
    counts = [len(r["solutions"]) for r in solved]
    if counts:
        print(f"IK solutions per pose               : {min(counts)} to {max(counts)}, mean {np.mean(counts):.1f}")
    for r in failed[:5]:
        print(f"  FAILED at servo deg {np.round(r['servo_deg'], 6).tolist()}: {describe_failure(r)}")
    print("PASS\n" if not failed else f"FAIL ({len(failed)} poses)\n")
    return failed


def random_state(rng):
    return rng.uniform(SERVO_LIMITS_DEG[:, 0], SERVO_LIMITS_DEG[:, 1])


def random_round_trips(n, seed, show_first=0):
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n):
        r = round_trip(random_state(rng), random_state(rng))
        results.append(r)
        if i < show_first:
            show(i + 1, r)
    return summarize(f"{n} random joint states (seed {seed})", results)


def limit_round_trips(per_case, seed):
    """Each joint exactly at its lower and upper limit, the rest random."""
    rng = np.random.default_rng(seed + 1)
    results = []
    for joint in range(6):
        for bound in SERVO_LIMITS_DEG[joint]:
            for _ in range(per_case):
                servo = random_state(rng)
                servo[joint] = bound
                results.append(round_trip(servo, random_state(rng)))
    return summarize(f"each joint at its lower / upper limit ({per_case} states each)", results)


def shoulder_singular_state(rng):
    """Random joint state with the wrist centre at radius exactly d4 from the
    J1 axis: its in-plane coordinate in frame 1, a2 cos th2 + a3 cos th23 +
    d5 sin th234, is solved to zero through joint 4."""
    while True:
        q = joint_map.servo_to_visual(random_state(rng))
        theta = q + OFFSETS
        u = A2 * math.cos(theta[1]) + A3 * math.cos(theta[1] + theta[2])
        if abs(u) <= D5:
            th234 = math.asin(-u / D5) if rng.random() < 0.5 else math.pi - math.asin(-u / D5)
            q[3] = math.radians(wrap180(math.degrees(th234 - theta[1] - theta[2] - OFFSETS[3])))
            return joint_map.visual_to_servo(q)


def th5_state(rng, th5):
    """Random joint state with the DH wrist angle th5 set exactly."""
    q = joint_map.servo_to_visual(random_state(rng))
    q[4] = th5 - OFFSETS[4]
    servo = joint_map.visual_to_servo(q)
    servo[4] = wrap180(servo[4])  # keep J5 inside its +-359.8 deg limits
    return servo


def near_limit_wrist_singular_state(rng):
    """Exact wrist singularity with J2 or J3 close to a limit: there only part
    of joint 6's free range keeps joints 2-4 inside their limits."""
    servo = th5_state(rng, rng.choice([0.0, math.pi]))
    joint = rng.choice([1, 2])
    servo[joint] = rng.choice([-1.0, 1.0]) * rng.uniform(0.97, 1.0) * SERVO_LIMITS_DEG[joint, 1]
    return servo


def singular_round_trips(per_case, seed):
    rng = np.random.default_rng(seed + 2)
    pi = math.pi
    cases = (
        # name, make a state, check joints too?, reference for check a, states x per_case
        ("wrist straight, th5 = 0", lambda: th5_state(rng, 0.0), True, None, 1),
        ("wrist straight, th5 = 180 deg", lambda: th5_state(rng, pi), True, None, 1),
        ("  same, J6 reference off by 90 deg", lambda: th5_state(rng, 0.0), False, "j6+90", 1),
        ("  same, J2/J3 near limit, J6 ref off", lambda: near_limit_wrist_singular_state(rng), False, "j6+90", 4),
        ("near it, th5 = +-0.9e-8 rad", lambda: th5_state(rng, rng.choice([-1, 1]) * 0.9e-8), False, None, 1),
        ("near it, th5 = 180 deg +- 0.9e-8", lambda: th5_state(rng, pi + rng.choice([-1, 1]) * 0.9e-8), False, None, 1),
        ("near it, th5 = +-1.1e-8 rad", lambda: th5_state(rng, rng.choice([-1, 1]) * 1.1e-8), False, None, 1),
        ("near it, th5 = 180 deg +- 1.1e-8", lambda: th5_state(rng, pi + rng.choice([-1, 1]) * 1.1e-8), False, None, 1),
        ("near it, |sin th5| = 1e-6", lambda: th5_state(rng, 1e-6), False, None, 1),
        ("elbow in line, J3 = 0", lambda: np.r_[random_state(rng)[:2], 0.0, random_state(rng)[3:]], False, None, 1),
        ("shoulder, wrist centre at radius d4", lambda: shoulder_singular_state(rng), False, None, 1),
    )
    print(f"==== Singular poses ({per_case} random states each, near-limit row {4 * per_case}) ====")
    failed = []
    for name, make, check_joints, ref_kind, weight in cases:
        results = []
        count = weight * per_case
        for _ in range(count):
            servo = make()
            ref = None
            if ref_kind == "j6+90":
                ref = servo.copy()
                ref[5] = ref[5] + 90.0 if ref[5] + 90.0 <= SERVO_LIMITS_DEG[5, 1] else ref[5] - 90.0
            results.append(round_trip(servo, random_state(rng), ref))
        bad = [r for r in results
               if not pose_ok(r) or (check_joints and r["joint_err_deg"] > JOINT_TOL_DEG)]
        failed += bad
        solved = [r for r in results if r["error"] is None]

        def worst(key):
            return max((r[key] for r in solved), default=float("nan"))

        print(f"{name:<36}: {count - len(bad)}/{count}  position {worst('pos_err_m'):.1e} m, "
              f"rotation {worst('rot_err_deg'):.1e} deg, joint difference {worst('joint_err_deg'):.1e} deg"
              + (" (checked)" if check_joints else ""))
        for r in bad[:2]:
            print(f"  FAILED at servo deg {np.round(r['servo_deg'], 6).tolist()}: {describe_failure(r)}")
    print("PASS\n" if not failed else f"FAIL ({len(failed)} poses)\n")
    return failed


def test_random_round_trip():
    assert not random_round_trips(n=1000, seed=0)


def test_joint_limit_round_trip():
    assert not limit_round_trips(per_case=20, seed=0)


def test_singular_round_trip():
    assert not singular_round_trips(per_case=50, seed=0)


def main():
    parser = argparse.ArgumentParser(description="Random joint angles -> FK -> IK -> compare.")
    parser.add_argument("--n", type=int, default=1000, help="number of random joint states")
    parser.add_argument("--seed", type=int, default=0, help="random seed (same seed, same angles)")
    parser.add_argument("--show", type=int, default=3, help="print the first N random samples in full")
    parser.add_argument("--per-case", type=int, default=50,
                        help="random states per joint-limit and singular case")
    args = parser.parse_args()

    failed = random_round_trips(args.n, args.seed, args.show)
    failed += limit_round_trips(max(1, args.per_case // 2), args.seed)
    failed += singular_round_trips(args.per_case, args.seed)
    print("ALL PASS" if not failed else f"FAILED: {len(failed)} poses")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
