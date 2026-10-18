"""
ik_from_pose.py
----------------
Pose of the tool tip, the flange or the camera -- in the robot base frame or
relative to an object -> the tool tip pose in the base frame and the joint
angles to send to the robot.

    target given for frame F (tool / flange / camera):
        in the base frame:      T_base_F
        relative to an object:  T_base_F = T_base_object @ T_object_F
            with T_base_object = T_base_camera(q_detect) @ T_camera_object  (FoundationPose)
              or T_base_object given directly
    -> tool tip pose:           T_base_tool = T_base_F @ inv(T_FLANGE_F) @ T_FLANGE_TOOL
    -> joint angles:            closed-form IK of T_base_tool (manipulator_kinematics)

Every answer is checked by putting it back through FK for the frame you
gave. Positions in meters. --yaw (with --xyz) turns the frame about the
vertical: + is counter-clockwise seen from above (about base +z, or about
the object's +z with --object).

Run:
    python3 ik_from_pose.py                                   # demo: home pose of all 3 frames + random points
    python3 ik_from_pose.py --frame camera --matrix T_base_camera.txt
    python3 ik_from_pose.py --frame flange --xyz -0.345 0.1175 0.4154
    python3 ik_from_pose.py --frame tool --xyz -0.40 0.10 0.05 --yaw 30 --table-z -0.05
  relative to an object (object pose from FoundationPose, camera frame):
    python3 ik_from_pose.py --object T_camera_object.txt --frame camera --xyz 0 0 0.30
    python3 ik_from_pose.py --object T_camera_object.txt --frame tool --matrix T_object_tool.txt
    python3 ik_from_pose.py --object T_base_object.txt --object-in base --frame tool --xyz 0 0 0.02
In code:
    from ik_from_pose import facing_pose, object_in_base, solve_frame
    T_base_object = object_in_base(T_camera_object)                  # FoundationPose, image at home
    T_base_tool, joints = solve_frame(T_base_object @ facing_pose(0, 0, 0.30), "camera")
    # joints[0]: send this (nearest home; pass current_servo_deg=... for the arm's joints)
"""

import argparse
import sys

import numpy as np

from manipulator_kinematics import (
    T_FLANGE_CAMERA, T_FLANGE_TOOL, IKError, TableConstraint, fk, ik_solutions,
    joint_map, pose, rot_x, rot_z,
)

FRAMES = ("tool", "flange", "camera")
FLANGE_TO = {"tool": T_FLANGE_TOOL, "flange": np.eye(4), "camera": T_FLANGE_CAMERA}


def to_tool_pose(T_base_frame, frame="tool"):
    """Pose of `frame` (tool / flange / camera) in the base frame -> the tool
    tip pose that puts that frame there. The three are bolted together, so
    this is one fixed transform."""
    return np.asarray(T_base_frame, float) @ np.linalg.inv(FLANGE_TO[frame]) @ T_FLANGE_TOOL


def object_in_base(T_object, object_in="camera", q_detect_servo_deg=None):
    """T_base_object from an object pose given in the camera frame (FoundationPose,
    taken at joints q_detect, default home) or already in the base frame."""
    T_object = np.asarray(T_object, float)
    if object_in == "base":
        return T_object
    q_detect = joint_map.home_servo_deg if q_detect_servo_deg is None else q_detect_servo_deg
    return fk(joint_map.servo_to_visual(q_detect), "camera") @ T_object


def solve_frame(T_base_frame, frame="tool", current_servo_deg=None, table=None):
    """Target pose of `frame` in the base frame -> (T_base_tool, joint
    solutions in servo deg, nearest the current joints first: send [0]).
    Raises IKError if nothing reaches it."""
    T_base_tool = to_tool_pose(T_base_frame, frame)
    current = joint_map.home_servo_deg if current_servo_deg is None else current_servo_deg
    return T_base_tool, ik_solutions(T_base_tool, current, table)


def down_pose(frame, x, y, z, yaw_deg=0.0):
    """`frame` at (x, y, z) in the base frame with its home orientation
    (pointing straight down), turned yaw_deg about the vertical."""
    return pose(rot_z(np.radians(yaw_deg)) @ fk(np.zeros(6), frame)[:3, :3], [x, y, z])


def facing_pose(x, y, z, yaw_deg=0.0):
    """A frame at (x, y, z) in the OBJECT frame with its z axis along the
    object's -z -- parallel to it, so it looks at the object's origin only when
    x = y = 0 and z > 0 -- and its x axis along the object's x, turned yaw_deg
    about the object's +z (+ = counter-clockwise seen from that side, the
    same sense as down_pose)."""
    return pose(rot_z(np.radians(yaw_deg)) @ rot_x(np.pi), [x, y, z])


def fk_errors(T_target, frame, servo_deg):
    """(position [m], max rotation-matrix element) error of FK of `frame`."""
    T = fk(joint_map.servo_to_visual(servo_deg), frame)
    return float(np.linalg.norm(T[:3, 3] - T_target[:3, 3])), float(np.abs(T[:3, :3] - T_target[:3, :3]).max())


def show(title, T_base_frame, frame, current_servo_deg=None, table=None, T_base_object=None):
    """Solve one target and print everything. Returns False if the target is
    unreachable or an FK check failed."""
    np.set_printoptions(precision=4, suppress=True)
    print(title)
    if T_base_object is not None:
        print(f"object pose T_base_object:\n{T_base_object}")
    print(f"{frame} target T_base_{frame}:\n{T_base_frame}")
    T_base_tool = to_tool_pose(T_base_frame, frame)
    if frame != "tool":
        print(f"-> tool tip pose T_base_tool:\n{T_base_tool}")
    try:
        _, solutions = solve_frame(T_base_frame, frame, current_servo_deg, table)
    except IKError as e:
        print(f"  not reachable: {e}")
        if np.abs(T_base_frame[:3, 3]).max() > 2.0:
            print("  (positions are in meters -- did you give millimeters?)")
        print()
        return False
    ok = True
    for k, s in enumerate(solutions, start=1):
        pos_err, rot_err = fk_errors(T_base_frame, frame, s)
        good = pos_err < 1e-9 and rot_err < 1e-9
        ok &= good
        mark = ("  <- send this one (nearest the current joints)" if k == 1 else "") if good \
            else "  FK CHECK FAILED - do not send"
        print(f"  {k}. servo deg {np.round(s, 6) + 0.0}   FK check ({frame}): {pos_err:.1e} m, {rot_err:.1e}{mark}")
    print()
    return ok


def random_points(n, seed, frame, yaw_deg, current_servo_deg=None, table=None):
    """n random positions of `frame` (pointing down) -> joint angles.
    Returns the number of reachable points whose FK check failed."""
    rng = np.random.default_rng(seed)
    print(f"==== {n} random {frame} positions (seed {seed}), pointing down ====")
    reached = bad = 0
    for i in range(1, n + 1):
        x, y = rng.uniform(-0.6, 0.6, 2)
        z = rng.uniform(-0.2, 0.5)
        T = down_pose(frame, x, y, z, yaw_deg)
        head = f"{i:3d}. x {x:+.3f}  y {y:+.3f}  z {z:+.3f} m"
        try:
            _, solutions = solve_frame(T, frame, current_servo_deg, table)
        except IKError as e:
            print(f"{head}  ->  not reachable ({str(e).split(':')[0]})")
            continue
        reached += 1
        pos_err, rot_err = fk_errors(T, frame, solutions[0])
        good = pos_err < 1e-9 and rot_err < 1e-9
        bad += not good
        angles = " ".join(f"{a:8.2f}" for a in solutions[0])
        print(f"{head}  ->  joints [{angles}] deg  ({len(solutions)} solutions, "
              f"FK check {pos_err:.0e} m){'' if good else '  FK CHECK FAILED'}")
    print(f"\n{reached}/{n} points reachable, {reached - bad}/{reached} pass the FK check\n")
    return bad


def load_matrix(path):
    """4x4 transform from a .npy, or a text file of 16 numbers: np.savetxt
    output, or a matrix pasted the way numpy prints it ([[...]], commas ok).
    The rotation part is snapped to the nearest exact rotation (it is usually
    rounded to a few decimals); a big correction or a mirror image is refused."""
    try:
        if path.endswith(".npy"):
            values = np.asarray(np.load(path), float).ravel()
        else:
            with open(path) as f:
                text = f.read().replace("[", " ").replace("]", " ").replace(",", " ")
            values = np.array(text.split(), float)
    except (OSError, ValueError) as e:
        sys.exit(f"{path}: cannot read a 4x4 matrix ({e})")
    if values.size != 16:
        sys.exit(f"{path}: expected 16 numbers (a 4x4 matrix), found {values.size}")
    T = values.reshape(4, 4)
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        sys.exit(f"{path}: the last row must be 0 0 0 1")
    U, _, Vt = np.linalg.svd(T[:3, :3])
    if np.linalg.det(U @ Vt) < 0:
        sys.exit(f"{path}: the rotation part is a mirror image (determinant -1), not a rotation")
    fix = float(np.abs(U @ Vt - T[:3, :3]).max())
    if fix > 1e-2:
        sys.exit(f"{path}: the rotation part is not a rotation matrix (off by {fix:.1e})")
    if fix > 1e-6:
        print(f"({path}: rotation snapped to the nearest exact rotation, correction {fix:.1e})")
    T = T.copy()
    T[:3, :3] = U @ Vt
    return T


def main():
    parser = argparse.ArgumentParser(description="Pose of tool / flange / camera (base frame or relative "
                                                 "to an object) -> tool tip pose and joint angles.")
    parser.add_argument("--frame", choices=FRAMES, default="tool", help="which frame the pose is for (default tool)")
    given = parser.add_mutually_exclusive_group()
    given.add_argument("--matrix", help="4x4 pose of --frame: .npy, or a text file of 16 numbers "
                                        "(np.savetxt output, or a matrix pasted as the scripts print it)")
    given.add_argument("--xyz", type=float, nargs=3, metavar=("X", "Y", "Z"),
                       help="position of --frame [m]; base frame: pointing straight down with its home "
                            "heading; with --object: in the object frame, z axis along the object's -z")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="with --xyz: turn about the vertical [deg], + = counter-clockwise seen from above")
    parser.add_argument("--object", help="object pose 4x4 file: --matrix / --xyz are then relative to the object")
    parser.add_argument("--object-in", choices=("camera", "base"), default="camera",
                        help="the --object pose is in the camera frame (FoundationPose, default) or the base frame")
    parser.add_argument("--detect", type=float, nargs=6, metavar="J",
                        help="joints when the camera image was taken, servo deg (default home; "
                             "only with --object in camera mode)")
    parser.add_argument("--current", type=float, nargs=6, metavar="J",
                        help="current joint angles, servo deg (default home)")
    parser.add_argument("--table-z", type=float, default=None, help="table height [m]: drop solutions below it")
    parser.add_argument("--n", type=int, default=10, help="number of random points in the demo")
    parser.add_argument("--seed", type=int, default=0, help="random seed for the demo")
    args = parser.parse_args()
    if args.yaw and not args.xyz:
        parser.error("--yaw only applies to --xyz")
    if args.detect is not None and not (args.object and args.object_in == "camera"):
        parser.error("--detect only applies to --object in camera mode")
    if args.table_z is not None and abs(args.table_z) > 2.0:
        parser.error("--table-z is in meters -- did you give millimeters?")
    table = TableConstraint(z=args.table_z) if args.table_z is not None else None

    if args.matrix or args.xyz:
        if args.object:
            T_base_object = object_in_base(load_matrix(args.object), args.object_in, args.detect)
            T_object_frame = load_matrix(args.matrix) if args.matrix else facing_pose(*args.xyz, args.yaw)
            T_base_frame = T_base_object @ T_object_frame
            title = f"{args.frame} given relative to the object ({args.matrix or args.xyz}):"
        else:
            T_base_object = None
            T_base_frame = load_matrix(args.matrix) if args.matrix else down_pose(args.frame, *args.xyz, args.yaw)
            title = f"{args.frame} given in the base frame ({args.matrix or args.xyz}):"
        ok = show(title, T_base_frame, args.frame, args.current, table, T_base_object)
        sys.exit(0 if ok else 2)
    if args.object:
        sys.exit("--object needs --matrix or --xyz: the pose of --frame relative to the object")

    # demo: the home pose of each frame must give the home joint angles
    ok = True
    for frame in FRAMES:
        T_home = fk(np.zeros(6), frame)
        ok &= show(f"==== home {frame} pose ====", T_home, frame, args.current, table)
        if args.current is None and table is None:
            ok &= bool(np.allclose(solve_frame(T_home, frame)[1][0], joint_map.home_servo_deg, atol=1e-6))
    bad = random_points(args.n, args.seed, args.frame, args.yaw, args.current, table)
    sys.exit(0 if ok and not bad else 1)


if __name__ == "__main__":
    main()
