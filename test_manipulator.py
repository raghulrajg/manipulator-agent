"""
test_manipulator.py
--------------------
Checks the manipulator code with no robot or camera: FK against Dobot's
official Nova 2 model and against the DH tables, the closed-form IK, the
table constraint, and a full detect -> plan pick.

Run: python3 test_manipulator.py
"""

import math

import numpy as np

from manipulator_agent import ManipulatorAgent, as_pose, pose_from_rvec_tvec
from manipulator_kinematics import (
    IKError, SERVO_LIMITS_DEG, T_FLANGE_TOOL, TableConstraint, build_dh_params, fk,
    forward_kinematics_6dof, ik_flange, ik_solutions, joint_map, pose, rot_x, rot_z,
    solve_ik, to_servo_near,
)

HOME = joint_map.home_servo_deg
rng = np.random.default_rng(7)

# Dobot's official Nova 2 model: joint origins (xyz [m], rpy [rad]) from
# dobot_description/urdf/nova2_robot.urdf in github.com/Dobot-Arm/TCP-IP-ROS-6AXis
# (commit 0924e785). Each joint then turns about its local z by the
# controller's joint angle as is -- Dobot's driver passes them straight in.
NOVA2_URDF_JOINTS = [
    ((0.0, 0.0, 0.2234), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, 1.5708, 0.0)),
    ((-0.28, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.22501, 0.0, 0.1175), (0.0, 0.0, -1.5708)),
    ((0.0, -0.12, 0.0), (1.5708, 0.0, 0.0)),
    ((0.0, 0.088004, 0.0), (-1.5708, 0.0, 0.0)),
]
R_FLANGE_TO_URDF_LINK6 = rot_z(-math.pi / 2)  # the two flange frames differ by this


def nova2_urdf_fk(servo_deg):
    """Flange (link6) pose of Dobot's URDF at controller angles servo_deg."""
    T = np.eye(4)
    for ((x, y, z), (roll, pitch, yaw)), angle in zip(NOVA2_URDF_JOINTS, servo_deg):
        R_pitch = np.array([[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0],
                            [-math.sin(pitch), 0.0, math.cos(pitch)]])
        T = T @ pose(rot_z(yaw) @ R_pitch @ rot_x(roll), [x, y, z]) @ pose(rot_z(math.radians(angle)), [0, 0, 0])
    return T


def table_fk(q, table):
    """FK straight from one DH table (no fixed flange offset)."""
    return forward_kinematics_6dof(build_dh_params(q, table))[0]


def random_servo_within_limits():
    return rng.uniform(SERVO_LIMITS_DEG[:, 0], SERVO_LIMITS_DEG[:, 1])


def test_fk_matches_dobot_nova2_and_dh_tables():
    print("=" * 60)
    print("FK vs Dobot's official Nova 2 model and the DH tables")
    print("=" * 60)
    worst_pos = worst_rot = 0.0
    for _ in range(500):
        servo = random_servo_within_limits()
        T, U = fk(joint_map.servo_to_visual(servo), "flange"), nova2_urdf_fk(servo)
        worst_pos = max(worst_pos, float(np.linalg.norm(T[:3, 3] - U[:3, 3])))
        R = (T[:3, :3] @ R_FLANGE_TO_URDF_LINK6).T @ U[:3, :3]
        worst_rot = max(worst_rot, math.degrees(math.acos(min(1.0, (np.trace(R) - 1) / 2))))
    # URDF a3 0.22501, d6 0.088004 and pi/2 written as 1.5708 explain the rest
    assert worst_pos < 5e-5 and worst_rot < 0.01, (worst_pos, worst_rot)

    for _ in range(300):
        q = rng.uniform(-np.pi, np.pi, 6)
        assert np.allclose(fk(q, "tool"), table_fk(q, "tool"), atol=1e-12)  # any joint 6
        q[5] = 0.0  # the camera table is exact with joint 6 at home
        assert np.allclose(fk(q, "camera"), table_fk(q, "camera"), atol=1e-12)
    assert np.allclose(T_FLANGE_TOOL, pose(np.eye(3), [0.0, 0.0, 0.197]), atol=1e-12)

    # With joint 6 turned, the camera's 76.5 mm sideways offset turns with
    # the flange; the d5-based camera table keeps it fixed, so they drift apart.
    q = np.radians([10, -20, 30, 15, -25, 90])
    drift = np.linalg.norm(fk(q, "camera")[:3, 3] - table_fk(q, "camera")[:3, 3])
    assert abs(drift - 2 * 0.0765 * math.sin(math.radians(45))) < 1e-9
    print(f"OK: flange FK matches Dobot's Nova 2 model within {worst_pos * 1000:.3f} mm / "
          f"{worst_rot:.4f} deg; tool equals its DH table (tip 0.197 m out on the flange axis); "
          f"camera equals its table at q6 = 0 and at q6 = 90 deg the d5 camera table is "
          f"{drift * 1000:.0f} mm off the flange-mounted camera.\n")


def test_ik_round_trip():
    print("=" * 60)
    print("Closed-form IK round trip (random joint states)")
    print("=" * 60)
    worst = 0.0
    for _ in range(2000):
        q = joint_map.servo_to_visual(random_servo_within_limits())
        T = fk(q, "tool")
        solutions = ik_flange(T @ np.linalg.inv(T_FLANGE_TOOL))
        assert solutions, f"no IK solution for a pose FK produced (q = {np.degrees(q)})"
        for s in solutions:
            worst = max(worst, float(np.abs(fk(s, "tool") - T).max()))
            assert np.all((s >= -np.pi) & (s < np.pi)), "ik_flange must wrap to [-pi, pi)"
        wrapped = [np.abs((s - q + np.pi) % (2 * np.pi) - np.pi).max() for s in solutions]
        assert min(wrapped) < 1e-7, "IK lost the joint state the pose came from"
    assert worst < 1e-9, worst

    T_home = fk(np.zeros(6), "tool")
    assert np.allclose(solve_ik(T_home, HOME), HOME)
    print(f"OK: every solution reproduces its pose (max error {worst:.1e}), the source joint "
          f"state is always among them, home solves to {HOME.tolist()}.\n")


def test_table_constraint_filters_ik():
    print("=" * 60)
    print("Table constraint: no IK solution may go below the table")
    print("=" * 60)
    table = TableConstraint(z=0.0)
    filtered_poses = 0
    for _ in range(4000):
        q = joint_map.servo_to_visual(random_servo_within_limits())
        T = fk(q, "tool")
        if T[2, 3] > 0.12 or table.violations(q):
            continue  # want reachable tool poses close above the table
        raw = [to_servo_near(s, HOME) for s in ik_flange(T @ np.linalg.inv(T_FLANGE_TOOL))]
        raw = [s for s in raw if s is not None]
        below = [s for s in raw if table.violations(joint_map.servo_to_visual(s))]
        usable = ik_solutions(T, HOME, table)
        for s in usable:
            assert not table.violations(joint_map.servo_to_visual(s))
        assert len(usable) <= len(raw) - len(below)
        filtered_poses += bool(below)
    assert filtered_poses > 20, "test never hit a pose with a below-table branch"

    T_under = pose(rot_x(math.pi), [-0.40, -0.10, -0.01])  # tip 1 cm into the table
    try:
        ik_solutions(T_under, HOME, table)
        raise AssertionError("a tool tip under the table was accepted")
    except IKError as e:
        refusal = str(e)
    assert "gripper" in refusal and "too low" in refusal, refusal
    print(f"OK: below-table branches dropped on {filtered_poses} poses; a tip 1 cm "
          f"into the table is refused:\n    {refusal}\n")


def test_pick_from_camera_detection():
    print("=" * 60)
    print("Pick: camera detection at home -> T_base_object -> constrained IK")
    print("=" * 60)
    agent = ManipulatorAgent()  # table = plane the object lies on
    T_base_camera = fk(np.zeros(6), "camera")
    for x, y, z, yaw_deg in ((-0.42, -0.12, 0.05, 0), (-0.48, -0.20, -0.05, 35),
                             (-0.33, 0.02, 0.0, -120), (-0.50, -0.05, -0.10, 80)):
        T_base_object_true = pose(rot_z(math.radians(yaw_deg)), [x, y, z])
        T_camera_object = np.linalg.inv(T_base_camera) @ T_base_object_true  # what the camera sees

        plan = agent.plan_pick(T_camera_object)
        assert np.allclose(plan.T_base_object, T_base_object_true, atol=1e-12)
        assert np.allclose(plan.T_object_base @ plan.T_base_object, np.eye(4), atol=1e-12)
        assert abs(plan.table_z - z) < 1e-12
        table = agent.table_for(plan.T_base_object)

        steps = round(agent.approach_height / agent.descent_step)
        names = [w.name for w in plan.waypoints]
        assert names == ["pre_grasp"] + ["descend"] * (steps - 1) + ["grasp"] + ["lift"] * steps, names
        for w in plan.waypoints:
            q = joint_map.servo_to_visual(w.q_servo_deg)
            assert np.allclose(fk(q, "tool"), w.T_base_tool, atol=1e-9)
            assert np.allclose(w.T_base_tool[:3, 2], [0, 0, -1], atol=1e-9), "tool must point down"
            assert np.allclose(w.T_base_tool[:2, 3], [x, y], atol=1e-9), "tip off the line above the object"
            assert not table.violations(q)
            assert np.all(w.q_servo_deg >= SERVO_LIMITS_DEG[:, 0])
            assert np.all(w.q_servo_deg <= SERVO_LIMITS_DEG[:, 1])
        grasp = plan.waypoints[steps]
        assert grasp.name == "grasp" and grasp.gripper == "close"
        assert np.allclose(grasp.T_base_tool[:3, 3], [x, y, z + agent.grasp_height], atol=1e-9)
        cos_yaw = abs(grasp.T_base_tool[:3, 0] @ T_base_object_true[:3, 0])
        assert abs(cos_yaw - 1.0) < 1e-9, "tool x not along the object's x"

        # Every joint move (from home on) stays above the table, and on the
        # way down / up the tip stays on the vertical line through the object.
        qs = [np.zeros(6)] + [joint_map.servo_to_visual(w.q_servo_deg) for w in plan.waypoints]
        drift = 0.0
        for i, (a, b) in enumerate(zip(qs, qs[1:])):
            assert not table.path_violations(a, b), f"move {i} goes below the table"
            if i > 0:
                for t in np.linspace(0.0, 1.0, 11):
                    tip_xy = fk(a + t * (b - a), "tool")[:2, 3]
                    drift = max(drift, float(np.linalg.norm(tip_xy - [x, y])))
        assert drift < 1e-3, f"tip drifts {drift * 1000:.2f} mm off the vertical line"
        print(f"  object ({x:+.2f}, {y:+.2f}, {z:+.2f}) yaw {yaw_deg:+4d} -> grasp servo deg "
              f"{np.round(grasp.q_servo_deg, 1).tolist()}, line drift {drift * 1000:.2f} mm")
    print("OK: object recovered in the base frame, tool tip on it pointing down, "
          "every waypoint and move above the table.\n")


def test_pick_refuses_to_go_below_table():
    print("=" * 60)
    print("Pick refused when the grasp would go into the table")
    print("=" * 60)
    T_base_object = pose(np.eye(3), [-0.42, -0.12, 0.0])
    T_camera_object = np.linalg.inv(fk(np.zeros(6), "camera")) @ T_base_object
    agent = ManipulatorAgent(table_z=0.03)  # table surface 3 cm above where the object was seen
    try:
        agent.plan_pick(T_camera_object)
        raise AssertionError("planned a grasp below the table")
    except IKError as e:
        refusal = str(e)
    assert "too low" in refusal, refusal
    print(f"OK: {refusal[:110]}...\n")


def test_detection_formats():
    print("=" * 60)
    print("Detection formats (cv2 rvec/tvec, 4x4)")
    print("=" * 60)
    T = pose_from_rvec_tvec([0.0, 0.0, math.pi / 2], [0.1, 0.2, 0.3])
    assert np.allclose(T, pose(rot_z(math.pi / 2), [0.1, 0.2, 0.3]))
    rvec = rng.normal(size=3)
    R = pose_from_rvec_tvec(rvec, [0, 0, 0])[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-12
    assert np.allclose(R @ rvec, rvec), "rotation axis must be rvec"
    assert np.allclose(as_pose({"rvec": [0, 0, math.pi / 2], "tvec": [0.1, 0.2, 0.3]}), T)
    assert np.allclose(as_pose(T.tolist()), T)
    print("OK\n")


if __name__ == "__main__":
    test_fk_matches_dobot_nova2_and_dh_tables()
    test_ik_round_trip()
    test_table_constraint_filters_ik()
    test_pick_from_camera_detection()
    test_pick_refuses_to_go_below_table()
    test_detection_formats()
