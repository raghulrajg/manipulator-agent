"""
fk_from_joints.py
------------------
Joint angles (servo degrees, as the controller reports them) -> forward
kinematics: the flange, tool tip and camera poses in the robot base frame.

How it is computed (manipulator_kinematics):
  1. servo deg -> model angles q [rad]:  q = direction * (servo - home_servo)
  2. DH table rows [alpha, a, d, q + theta_offset]
  3. multiply the six joint transforms: T_0^6 = A1 @ A2 @ ... @ A6  (flange)
  4. tool / camera = flange @ fixed flange->tool / flange->camera offset

Run:
    python3 fk_from_joints.py                                   # home pose
    python3 fk_from_joints.py --joints -180 0 -90 0 90 90       # given servo angles
    python3 fk_from_joints.py --joints -150 10 -80 5 90 120 --steps   # also every joint frame
    python3 fk_from_joints.py --random 5 --seed 1               # random joint angles
"""

import argparse

import numpy as np

from manipulator_kinematics import (
    SERVO_LIMITS_DEG, build_dh_params, fk, forward_kinematics_6dof, joint_map,
)


def forward_kinematics(servo_deg):
    """Servo angles [deg] -> {"flange", "tool", "camera"}: 4x4 poses in the
    base frame, meters."""
    q = joint_map.servo_to_visual(servo_deg)
    return {frame: fk(q, frame) for frame in ("flange", "tool", "camera")}


def show(servo_deg, steps=False):
    servo_deg = np.asarray(servo_deg, dtype=float)
    print(f"Joint angles (servo deg): {servo_deg}")
    print(f"Model angles q (deg):     {np.degrees(joint_map.servo_to_visual(servo_deg))}")
    if steps:
        _, transforms, cumulative = forward_kinematics_6dof(build_dh_params(joint_map.servo_to_visual(servo_deg)))
        for i, (A, T) in enumerate(zip(transforms, cumulative), start=1):
            print(f"\nT_{i-1}^{i} (joint {i}):\n{A}\nT_0^{i}:\n{T}")
        print()
    for frame, T in forward_kinematics(servo_deg).items():
        print(f"\n{frame} pose T_base_{frame}:\n{T}")
        print(f"  position (x, y, z) m: {T[:3, 3]}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Joint angles -> flange / tool / camera pose (forward kinematics).")
    parser.add_argument("--joints", type=float, nargs=6, metavar="J", help="servo angles J1..J6 [deg]")
    parser.add_argument("--random", type=int, default=0, help="number of random joint states")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--steps", action="store_true", help="also print every joint transform")
    args = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)

    if args.random:
        rng = np.random.default_rng(args.seed)
        for _ in range(args.random):
            show(rng.uniform(SERVO_LIMITS_DEG[:, 0], SERVO_LIMITS_DEG[:, 1]), args.steps)
    else:
        show(joint_map.home_servo_deg if args.joints is None else args.joints, args.steps)


if __name__ == "__main__":
    main()
