"""
manipulator_agent.py
---------------------
Pick planning for the arm: where the object is in the arm's base frame, and
the joint angles that bring the tool tip down onto it from above without
any part of the arm going below the table.

1. Object in the base frame. The wrist camera sees the object from the home
   pose -- from there it looks straight down at the table the object lies
   on -- so:

       T_base_object = T_base_camera(q_detect) @ T_camera_object

   T_base_camera is the camera table's FK at the joint state the image was
   taken at (home by default) and T_camera_object is your detector's output
   (cv2.aruco / solvePnP rvec + tvec -> pose_from_rvec_tvec). The base as
   seen from the object is the inverse, PickPlan.T_object_base.

2. Grasp poses. Tool z straight down into the table, tool x along the
   object's yaw (+ grasp_yaw_offset_deg), tip grasp_height above the
   object's origin; a pre-grasp approach_height above that, which is also
   where it lifts back to. A parallel gripper grips the same way turned
   180 deg, so both are tried and the one with less joint travel wins.

3. IK with the table constraint (manipulator_kinematics.ik_solutions): each
   waypoint's solution must keep the whole arm above the table, and so must
   the joint moves between waypoints. The descent and the lift follow the
   vertical line through the object as joint waypoints every descent_step,
   all on one IK branch (no wrist or elbow flip above the object): a
   single joint move from pre-grasp to grasp would bow sideways, up to
   13 mm over 10 cm near the edge of the workspace.

The table height is table_z if given, otherwise the plane the object's frame
sits on: the ArUco markers lie flat on the table, so that is the table
surface the camera found at home.

Run a quick plan from a camera detection (x y z [m], yaw [deg]):
    python3 manipulator_agent.py 0.02 -0.03 0.30 20
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from manipulator_kinematics import (
    IKError, TableConstraint, fk, ik_solutions, joint_map, pose, rot_x, rot_z,
)


def pose_from_rvec_tvec(rvec, tvec):
    """OpenCV rvec (axis-angle, Rodrigues) + tvec -> 4x4 pose, without
    needing cv2 here."""
    r = np.asarray(rvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(r))
    R = np.eye(3)
    if angle > 1e-12:
        k = r / angle
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        R = R + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return pose(R, np.asarray(tvec, dtype=float).reshape(3))


def as_pose(value):
    """A pose from JSON: a 4x4 nested list, or {"rvec": [...], "tvec": [...]}
    straight from cv2.aruco / solvePnP."""
    if isinstance(value, dict):
        return pose_from_rvec_tvec(value["rvec"], value["tvec"])
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"expected a 4x4 pose or {{rvec, tvec}}, got shape {T.shape}")
    return T


@dataclass
class Waypoint:
    name: str                 # "pre_grasp" | "descend" | "grasp" | "lift"
    T_base_tool: np.ndarray   # tool tip pose at this waypoint
    q_servo_deg: np.ndarray   # joint target to command, servo degrees
    gripper: Optional[str]    # "open" / "close" once there, or None


@dataclass
class PickPlan:
    T_base_object: np.ndarray
    table_z: float
    waypoints: List[Waypoint]

    @property
    def T_object_base(self):
        return np.linalg.inv(self.T_base_object)

    def to_dict(self):
        """JSON-friendly summary, e.g. for logging or for your arm controller."""
        return {
            "T_base_object": np.round(self.T_base_object, 5).tolist(),
            "table_z": round(float(self.table_z), 5),
            "waypoints": [
                {
                    "name": w.name,
                    "q_servo_deg": np.round(w.q_servo_deg, 3).tolist(),
                    "tool_xyz": np.round(w.T_base_tool[:3, 3], 5).tolist(),
                    "gripper": w.gripper,
                }
                for w in self.waypoints
            ],
        }


class ManipulatorAgent:
    def __init__(self, table_z: Optional[float] = None,
                 link_clearance: float = 0.05, tip_clearance: float = 0.005,
                 table_x_range=None, table_y_range=None,
                 grasp_height: float = 0.02, approach_height: float = 0.10,
                 descent_step: float = 0.02, grasp_yaw_offset_deg: float = 0.0,
                 max_branch_jump_deg: float = 90.0):
        """
        table_z            table surface height in the arm base frame [m];
                           None = the plane the detected object sits on
        link_clearance     how far elbow / wrist / camera stay above it [m]
        tip_clearance      how far the gripper tip stays above it [m]
        table_x/y_range    the table's footprint in the base frame (see
                           TableConstraint); None = endless
        grasp_height       tool tip above the object's origin at the grasp [m]
        approach_height    pre-grasp / lift height above the grasp [m]
        descent_step       spacing of the waypoints on the way down / up [m]
        grasp_yaw_offset_deg  tool x relative to the object's x about the
                           vertical, so the fingers close across the object
                           (tool x is the flange's x axis with your updated
                           tool table; use 90 if the fingers close along the
                           flange's y)
        max_branch_jump_deg   largest joint change allowed between descent
                           waypoints; more means the IK switched branch
        """
        self.table_z = table_z
        self.link_clearance = link_clearance
        self.tip_clearance = tip_clearance
        self.table_x_range = table_x_range
        self.table_y_range = table_y_range
        self.grasp_height = grasp_height
        self.approach_height = approach_height
        self.descent_step = descent_step
        self.grasp_yaw_offset_deg = grasp_yaw_offset_deg
        self.max_branch_jump_deg = max_branch_jump_deg

    def object_in_base(self, T_camera_object, q_detect_servo_deg=None):
        """T_base_object = T_base_camera(q_detect) @ T_camera_object, with
        q_detect the servo angles the image was taken at (default: home)."""
        q_detect = joint_map.home_servo_deg if q_detect_servo_deg is None else q_detect_servo_deg
        T_base_camera = fk(joint_map.servo_to_visual(q_detect), "camera")
        return T_base_camera @ np.asarray(T_camera_object, dtype=float)

    def table_for(self, T_base_object) -> TableConstraint:
        z = self.table_z if self.table_z is not None else float(T_base_object[2, 3])
        return TableConstraint(z, self.link_clearance, self.tip_clearance,
                               self.table_x_range, self.table_y_range)

    def grasp_poses(self, T_base_object):
        """(T_pre_grasp, T_grasp) tool tip poses for both 180-deg-symmetric
        grips: tool z straight down, tool x along the object's yaw."""
        x_axis = T_base_object[:3, 0]
        yaw = math.atan2(x_axis[1], x_axis[0]) + math.radians(self.grasp_yaw_offset_deg)
        tip = T_base_object[:3, 3] + np.array([0.0, 0.0, self.grasp_height])
        above = tip + np.array([0.0, 0.0, self.approach_height])
        for psi in (yaw, yaw + math.pi):
            R = rot_z(psi) @ rot_x(math.pi)
            yield pose(R, above), pose(R, tip)

    def plan_pick(self, T_camera_object, q_detect_servo_deg=None,
                  q_start_servo_deg=None) -> PickPlan:
        """
        Plan pre-grasp -> descend -> grasp -> lift for an object detected at
        q_detect_servo_deg (default home), starting from q_start_servo_deg
        (default: where the detection was taken). Raises IKError, naming
        what was in the way, if no plan keeps the arm above the table.
        """
        q_detect = np.asarray(joint_map.home_servo_deg if q_detect_servo_deg is None
                              else q_detect_servo_deg, dtype=float)
        q_start = q_detect if q_start_servo_deg is None else np.asarray(q_start_servo_deg, dtype=float)
        T_base_object = self.object_in_base(T_camera_object, q_detect)
        table = self.table_for(T_base_object)

        start_bad = table.violations(joint_map.servo_to_visual(q_start))
        if start_bad:
            raise IKError("start pose is already below the table (" + ", ".join(start_bad)
                          + ") -- set table_x_range / table_y_range if that area isn't over the table")

        best, best_cost, errors = None, math.inf, []
        for T_pre, T_grasp in self.grasp_poses(T_base_object):
            try:
                grasps = ik_solutions(T_grasp, q_start, table)
            except IKError as e:
                errors.append(f"grasp: {e}")
                continue
            for q_grasp in grasps:
                try:
                    line = self._vertical_line(table, T_pre, T_grasp, q_grasp)
                except IKError as e:
                    errors.append(f"descent: {e}")
                    continue
                bad = table.path_violations(joint_map.servo_to_visual(q_start),
                                            joint_map.servo_to_visual(line[0][1]))
                if bad:
                    errors.append("start -> pre-grasp goes below the table: " + ", ".join(bad))
                    continue
                # start -> pre-grasp, then down the line and back up it
                qs = [q for _, q in line]
                cost = np.linalg.norm(qs[0] - q_start) + 2 * sum(
                    np.linalg.norm(b - a) for a, b in zip(qs, qs[1:]))
                if cost < best_cost:
                    best_cost, best = cost, line

        if best is None:
            raise IKError(f"cannot pick the object at {np.round(T_base_object[:3, 3], 4).tolist()} "
                          f"(table z = {table.z:.4f}): " + " | ".join(dict.fromkeys(errors)))
        (T_pre, q_pre), *descend, (T_grasp, q_grasp) = best
        waypoints = ([Waypoint("pre_grasp", T_pre, q_pre, "open")]
                     + [Waypoint("descend", T, q, None) for T, q in descend]
                     + [Waypoint("grasp", T_grasp, q_grasp, "close")]
                     + [Waypoint("lift", T, q, None) for T, q in reversed(best[:-1])])
        return PickPlan(T_base_object, table.z, waypoints)

    def _vertical_line(self, table, T_pre, T_grasp, q_grasp):
        """[(T_base_tool, q_servo_deg)] from the pre-grasp down to the grasp,
        one every descent_step, solved upward from q_grasp so the whole line
        stays on its IK branch. Raises IKError if a step can't be solved,
        flips branch, or its joint move goes below the table."""
        steps = max(1, math.ceil(self.approach_height / self.descent_step - 1e-9))
        line = [(T_grasp, q_grasp)]
        for k in range(1, steps + 1):
            T = T_grasp.copy()
            T[2, 3] += (T_pre[2, 3] - T_grasp[2, 3]) * k / steps
            q_below = line[-1][1]
            q = ik_solutions(T, q_below, table)[0]
            jump = float(np.abs(q - q_below).max())
            if jump > self.max_branch_jump_deg:
                raise IKError(f"would switch IK branch ({jump:.0f} deg joint jump)")
            bad = table.path_violations(joint_map.servo_to_visual(q_below),
                                        joint_map.servo_to_visual(q), steps=5)
            if bad:
                raise IKError("goes below the table: " + ", ".join(bad))
            line.append((T, q))
        return line[::-1]


def plan_place(agent, table, T_place_tool, q_start):
    """Put the object down with the grasp it is held in: tool tip at
    T_place_tool, approached straight down from approach_height above.
    Returns [(name, T_base_tool, servo_deg)] pre -> place."""
    T_pre = T_place_tool.copy()
    T_pre[2, 3] += agent.approach_height
    errors = []
    for q_place in ik_solutions(T_place_tool, q_start, table):
        try:
            line = agent._vertical_line(table, T_pre, T_place_tool, q_place)  # straight line, one IK branch
        except IKError as e:
            errors.append(str(e))
            continue
        bad = table.path_violations(joint_map.servo_to_visual(q_start), joint_map.servo_to_visual(line[0][1]))
        if bad:
            errors.append("carry move goes below the table: " + ", ".join(bad))
            continue
        names = ["pre_place"] + ["lower"] * (len(line) - 2) + ["place"]
        return [(n, T, q) for n, (T, q) in zip(names, line)]
    raise IKError("cannot put the object down at "
                  f"{np.round(T_place_tool[:3, 3], 4).tolist()}: " + " | ".join(dict.fromkeys(errors)))


def main():
    parser = argparse.ArgumentParser(
        description="Plan a top-down pick from an object detected by the wrist camera at home.")
    parser.add_argument("x", type=float, nargs="?", default=0.0, help="object x in the camera frame [m]")
    parser.add_argument("y", type=float, nargs="?", default=0.0, help="object y in the camera frame [m]")
    parser.add_argument("z", type=float, nargs="?", default=0.30,
                        help="object z in the camera frame (distance along the view axis) [m]")
    parser.add_argument("yaw", type=float, nargs="?", default=0.0,
                        help="object yaw about the camera's view axis [deg]")
    parser.add_argument("--table-z", type=float, default=None,
                        help="table height in the arm base frame [m] (default: the object's plane)")
    args = parser.parse_args()
    np.set_printoptions(precision=4, suppress=True)

    # Object lying on the table under a downward-looking camera: its z axis
    # points back up at the camera (the ArUco convention).
    T_camera_object = pose(rot_z(math.radians(args.yaw)) @ rot_x(math.pi), [args.x, args.y, args.z])
    agent = ManipulatorAgent(table_z=args.table_z)

    print(f"Camera at home (T_base_camera):\n{fk(np.zeros(6), 'camera')}\n")
    print(f"Detection (T_camera_object):\n{T_camera_object}\n")
    T_base_object = agent.object_in_base(T_camera_object)
    print(f"Object in the base frame (T_base_object = T_base_camera @ T_camera_object):\n{T_base_object}\n")
    print(f"Base in the object frame (T_object_base):\n{np.linalg.inv(T_base_object)}\n")

    try:
        plan = agent.plan_pick(T_camera_object)
    except IKError as e:
        print(f"No safe pick: {e}")
        return
    print(f"Table surface z = {plan.table_z:.4f} m (base frame)\n")
    for w in plan.waypoints:
        then = f"  then {w.gripper} the gripper" if w.gripper else ""
        print(f"{w.name:<9}  servo deg {w.q_servo_deg}  tool tip {w.T_base_tool[:3, 3]}{then}")


if __name__ == "__main__":
    main()
