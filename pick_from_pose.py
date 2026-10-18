"""
pick_from_pose.py
------------------
Full 6-DoF object pose (4x4 from FoundationPose, position AND orientation)
-> joint angles for an approach and a grasp along the object's own axes.

    T_base_object = T_base_camera(q_detect) @ T_camera_object     (FoundationPose output)
    T_base_grasp  = T_base_object @ T_OBJECT_GRASP                (where the tool tip goes)
    T_base_pre    = T_base_grasp @ Trans(0, 0, -approach)         (back off along tool z)

The tool comes in along its own z axis to the grasp, through joint
waypoints every 2 cm along that line. With table_z (--table-z) every
waypoint and every joint move between them, from the current joints on, is
checked against the table; without it there is no table check.

FoundationPose returns the pose in the camera's optical frame (OpenCV:
z forward, x right, y down) in meters. It has to match the camera DH
frame's axes; if your x/y come out turned, adjust the camera table's J6
offset in manipulator_kinematics.py.

Run:
    python3 pick_from_pose.py --pose pose.txt                # 4x4 saved with np.savetxt
    python3 pick_from_pose.py --pose pose.npy --table-z -0.05
    python3 pick_from_pose.py                                # demo: a tilted object
In code:
    from pick_from_pose import plan_from_pose
    plan = plan_from_pose(T_camera_object)                   # your FoundationPose 4x4
"""

import argparse
import sys

import numpy as np

from manipulator_kinematics import (
    IKError, TableConstraint, fk, ik_solutions, joint_map, pose, rot_x, rot_z,
)

# Where the tool tip goes, expressed in the OBJECT frame of your FoundationPose
# mesh. Default: 2 cm above the object origin along its +z, tool z pointing
# back down -z, tool x along the object's x. With your updated tool table
# tool x is the flange's x axis (your old table's tool x was the flange's y),
# so check which flange axis your fingers close along; if it is the flange's
# y, use pose(rot_x(np.pi) @ rot_z(np.pi / 2), [0.0, 0.0, 0.02]). Edit for
# your object.
T_OBJECT_GRASP = pose(rot_x(np.pi), [0.0, 0.0, 0.02])
APPROACH = 0.10   # pre-grasp distance back along the tool z [m]
STEP = 0.02       # waypoint spacing on the approach line [m]


def check_move(table, q_from_servo, q_to_servo):
    """Raise IKError if the joint move between two servo states dips below
    the table (no-op without a table)."""
    if table is not None:
        bad = table.path_violations(joint_map.servo_to_visual(q_from_servo),
                                    joint_map.servo_to_visual(q_to_servo))
        if bad:
            raise IKError("joint move goes below the table: " + ", ".join(bad))


def plan_from_pose(T_camera_object, q_detect_servo_deg=None, q_current_servo_deg=None,
                   table_z=None, grasp=T_OBJECT_GRASP, approach=APPROACH, symmetric=True):
    """
    FoundationPose 4x4 (camera frame) -> dict with T_base_object, the grasp
    pose and joint waypoints [(name, T_base_tool, servo_deg)] from pre-grasp
    down to grasp. q_detect: joints when the image was taken (default home);
    q_current: joints now (default q_detect). symmetric=True also tries the
    grasp turned 180 deg about the tool z (parallel gripper) and keeps the
    one with less joint motion. Raises IKError if nothing is safe.
    """
    q_detect = joint_map.home_servo_deg if q_detect_servo_deg is None else np.asarray(q_detect_servo_deg, float)
    q_now = q_detect if q_current_servo_deg is None else np.asarray(q_current_servo_deg, float)
    T_base_object = fk(joint_map.servo_to_visual(q_detect), "camera") @ np.asarray(T_camera_object, float)
    table = TableConstraint(z=table_z) if table_z is not None else None

    best, errors = None, []
    for turn in ((0.0, np.pi) if symmetric else (0.0,)):
        T_grasp = T_base_object @ grasp @ pose(rot_z(turn), [0, 0, 0])
        n = max(1, int(np.ceil(approach / STEP - 1e-9)))
        line = [T_grasp @ pose(np.eye(3), [0, 0, -approach * k / n]) for k in range(n, -1, -1)]
        try:
            q = ik_solutions(line[0], q_now, table)[0]           # pre-grasp, nearest current joints
            check_move(table, q_now, q)
            waypoints = [("pre_grasp", line[0], q)]
            for k, T in enumerate(line[1:], start=1):
                q_next = ik_solutions(T, q, table)[0]
                if np.abs(q_next - q).max() > 90:
                    raise IKError("approach line would switch IK branch")
                check_move(table, q, q_next)
                q = q_next
                waypoints.append(("grasp" if k == n else "approach", T, q))
        except IKError as e:
            errors.append(str(e))
            continue
        cost = np.linalg.norm(waypoints[0][2] - q_now) + sum(
            np.linalg.norm(b[2] - a[2]) for a, b in zip(waypoints, waypoints[1:]))
        if best is None or cost < best[0]:
            best = (cost, T_grasp, waypoints)
    if best is None:
        raise IKError("no safe grasp: " + " | ".join(dict.fromkeys(errors)))
    return {"T_base_object": T_base_object, "T_base_grasp": best[1], "waypoints": best[2]}


def load_pose(path):
    T = np.load(path) if path.endswith(".npy") else np.loadtxt(path)
    T = np.asarray(T, float).reshape(4, 4)
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
        sys.exit(f"{path}: the 3x3 part is not a rotation matrix")
    return T


def main():
    parser = argparse.ArgumentParser(description="FoundationPose 4x4 -> approach + grasp joint angles.")
    parser.add_argument("--pose", help="4x4 T_camera_object file (.txt or .npy), meters")
    parser.add_argument("--detect", type=float, nargs=6, metavar="J", help="joints when the image was taken (default home)")
    parser.add_argument("--current", type=float, nargs=6, metavar="J", help="joints now (default: --detect)")
    parser.add_argument("--table-z", type=float, default=None, help="table height in the base frame [m]")
    args = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)

    if args.pose:
        T_camera_object = load_pose(args.pose)
    else:  # demo: object 35 cm in front of the camera, tilted 20 deg about x and turned 30 deg
        T_camera_object = pose(rot_z(np.radians(30)) @ rot_x(np.pi + np.radians(20)), [0.02, -0.03, 0.35])
        print("(demo pose -- pass --pose your_pose.txt for a FoundationPose result)\n")

    print(f"T_camera_object (FoundationPose):\n{T_camera_object}\n")
    try:
        plan = plan_from_pose(T_camera_object, args.detect, args.current, args.table_z)
    except IKError as e:
        sys.exit(f"Cannot pick: {e}")
    print(f"T_base_object:\n{plan['T_base_object']}\n")
    print(f"T_base_grasp (tool tip target):\n{plan['T_base_grasp']}\n")
    for name, T, q in plan["waypoints"]:
        err = np.linalg.norm(fk(joint_map.servo_to_visual(q), "tool")[:3, 3] - T[:3, 3])
        print(f"{name:<9} servo deg {q}  tool tip {T[:3, 3]}  (FK check {err:.0e} m)")
    print("\nSend the waypoints in order (joint moves), close the gripper at 'grasp', "
          "then send them back in reverse to lift.")


if __name__ == "__main__":
    main()
