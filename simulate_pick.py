"""
simulate_pick.py
-----------------
3D simulation of the arm picking up the object and placing it somewhere
else, built on the same kinematics, IK and pick planner the robot uses
(numpy + matplotlib, no physics engine).

Sequence: home (the camera sees the object) -> pre-grasp -> straight
descent -> close the gripper -> lift -> carry to the place spot -> lower ->
open -> lift -> home. Every frame is the arm's own FK at the planned joint
angles; moves between waypoints are linear in joint space (like Dobot MovJ).

Before anything is drawn, the whole motion is checked at 1 deg steps: no
part of the arm below the table, the carried object never through the
table, and the object put down exactly at the place spot (position and
orientation). Any failure stops with a message saying what and where.

The object frame is its bottom centre with +z up -- what an ArUco board
gives. A FoundationPose mesh frame usually differs: move its origin to the
bottom centre (and z up) before passing it with --pose.

Run:
    python3 simulate_pick.py                                   # demo, saves pick.gif
    python3 simulate_pick.py --object -0.45 0.05 0.0 30        # object x y z [m] + yaw [deg], base frame
    python3 simulate_pick.py --pose T_camera_object.txt        # 4x4 object pose, camera frame, image at home
    python3 simulate_pick.py --place -0.25 0.40 --save pick.mp4
    python3 simulate_pick.py --show                            # open a window instead of saving
"""

import argparse
import math
import sys

import numpy as np

from manipulator_agent import ManipulatorAgent, plan_place
from manipulator_kinematics import (
    IKError, build_dh_params, fk, forward_kinematics_6dof, joint_map, pose, rot_z,
)
from pick_from_pose import load_pose

OBJECT_SIZE = (0.04, 0.04, 0.05)   # box x, y, z [m]; its frame origin is the bottom centre
FINGER_LEN = 0.045                 # gripper finger length [m]
OPEN_WIDTH = 0.08                  # finger gap when open [m]
DEG_PER_FRAME = 3.0                # animation: at most this much joint motion ...
M_PER_FRAME = 0.01                 # ... and this much tool tip travel per frame
CHECK_DEG = 1.0                    # safety check resolution along every move


def arm_points(servo_deg):
    """Base, joint origins 1-6 (flange last), tool tip and camera, base frame."""
    q = joint_map.servo_to_visual(servo_deg)
    _, _, cumulative = forward_kinematics_6dof(build_dh_params(q, "flange"))
    chain = [np.zeros(3)] + [T[:3, 3] for T in cumulative]
    return np.array(chain), fk(q, "tool"), fk(q, "camera")


def box_corners(T_base_object, size=OBJECT_SIZE):
    sx, sy, sz = size
    corners = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (0.0, sz)])
    return (T_base_object[:3, :3] @ corners.T).T + T_base_object[:3, 3]


def box_edges(T_base_object, size=OBJECT_SIZE):
    world = box_corners(T_base_object, size)
    return [(world[a], world[b]) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def rotation_angle_deg(R_a, R_b):
    R = R_a.T @ R_b
    s = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return math.degrees(math.atan2(s, (np.trace(R) - 1) / 2))


def plan(T_base_object, T_base_place, table_z):
    """Pick at T_base_object, place at T_base_place (same orientation, so the
    grasp carries over). Returns frames, table, waypoints, problems; a frame
    is dict(servo, gap, carrying, label, T_obj)."""
    home = joint_map.home_servo_deg
    agent = ManipulatorAgent(table_z=table_z)
    pick = agent.plan_pick(np.linalg.inv(fk(np.zeros(6), "camera")) @ T_base_object)
    table = agent.table_for(T_base_object)
    grasp = next(w for w in pick.waypoints if w.name == "grasp")
    T_tool_obj = np.linalg.inv(grasp.T_base_tool) @ T_base_object      # how the object is held
    try:
        place = plan_place(agent, table, T_base_place @ np.linalg.inv(T_tool_obj), pick.waypoints[-1].q_servo_deg)
    except IKError as e:
        raise IKError(f"{e} (place spot {np.round(T_base_place[:3, 3], 4).tolist()})") from None

    width = min(OBJECT_SIZE[0], OPEN_WIDTH)
    frames, problems = [], []
    state = {"q": home, "T_obj": T_base_object}

    def frame(q, gap, carrying, label):
        T_tool = fk(joint_map.servo_to_visual(q), "tool")
        if carrying:
            state["T_obj"] = T_tool @ T_tool_obj
        frames.append(dict(servo=q, gap=gap, carrying=carrying, label=label, T_obj=state["T_obj"]))

    def move_to(q, gap, carrying, label):
        a, b = state["q"], np.asarray(q, float)
        # safety check of the whole move at CHECK_DEG resolution, arm and carried object
        n_check = max(10, math.ceil(np.abs(b - a).max() / CHECK_DEG))
        tips = []
        for t in np.linspace(0.0, 1.0, n_check + 1):
            qv = joint_map.servo_to_visual(a + t * (b - a))
            bad = table.violations(qv)
            if bad:
                problems.append(f"{label}: arm below the table ({', '.join(bad)}) at {t:.0%} of the move")
            T_tool = fk(qv, "tool")
            tips.append(T_tool[:3, 3])
            if carrying:
                low = box_corners(T_tool @ T_tool_obj)[:, 2].min() - table.z
                if low < -1e-9:
                    problems.append(f"{label}: carried object {-low * 100:.1f} cm into the table at {t:.0%} of the move")
        travel = float(np.sum(np.linalg.norm(np.diff(tips, axis=0), axis=1)))
        steps = max(2, math.ceil(np.abs(b - a).max() / DEG_PER_FRAME), math.ceil(travel / M_PER_FRAME))
        for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
            frame(a + t * (b - a), gap, carrying, label)
        state["q"] = b

    def grip(start, end, carrying, label):
        for g in np.linspace(start, end, 8):
            frame(state["q"], g, carrying, label)

    for _ in range(12):
        frame(home, OPEN_WIDTH, False, "home: camera detects the object")
    labels = {"pre_grasp": "move above the object", "descend": "straight down", "grasp": "straight down",
              "lift": "lift"}
    for wp in pick.waypoints:
        carrying = wp.name == "lift"
        move_to(wp.q_servo_deg, width if carrying else OPEN_WIDTH, carrying, labels[wp.name])
        if wp.name == "grasp":
            grip(OPEN_WIDTH, width, False, "close the gripper")
            frames[-1].update(carrying=True)                     # holding it from here
    labels = {"pre_place": "carry to the place spot", "lower": "lower", "place": "lower"}
    for name, _, q in place:
        move_to(q, width, True, labels[name])
    grip(width, OPEN_WIDTH, True, "open the gripper")
    frames[-1].update(carrying=False)                            # released here
    for name, _, q in reversed(place[:-1]):
        move_to(q, OPEN_WIDTH, False, "lift away")
    move_to(home, OPEN_WIDTH, False, "back home")
    frames += [dict(frames[-1]) for _ in range(10)]

    T_end = frames[-1]["T_obj"]
    off = float(np.linalg.norm(T_end[:3, 3] - T_base_place[:3, 3]))
    turn = rotation_angle_deg(T_end[:3, :3], T_base_place[:3, :3])
    if off > 1e-6 or turn > 1e-6:
        problems.append(f"object put down {off * 1000:.2f} mm / {turn:.2f} deg away from the place spot")
    waypoints = [("pick", w.name, w.q_servo_deg) for w in pick.waypoints] + [("place", n, q) for n, _, q in place]
    return frames, table, waypoints, problems


def animate(frames, table, T_base_place, out, show, fps):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    zt = table.z
    tips = np.array([arm_points(f["servo"])[1][:3, 3] for f in frames])
    pts = np.vstack([np.vstack([arm_points(f["servo"])[0], box_corners(f["T_obj"])]) for f in frames[::3]]
                    + [box_corners(T_base_place), [[0.0, 0.0, 0.0]]])
    lo, hi = pts.min(axis=0) - 0.08, pts.max(axis=0) + 0.08
    lo[2] = min(lo[2], zt) - 0.02
    tx, ty = np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]])
    ring = np.linspace(0, 2 * np.pi, 30)

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9)

    def draw(i):
        f = frames[i]
        chain, T_tool, T_cam = arm_points(f["servo"])
        ax.cla()
        ax.plot_surface(tx, ty, np.full_like(tx, zt), color="#c9b79c", alpha=0.35, linewidth=0)
        ax.plot(0.08 * np.cos(ring), 0.08 * np.sin(ring), np.zeros_like(ring), color="#555555", lw=1.5)
        ax.plot(*chain.T, color="#3b6fb6", lw=6, solid_capstyle="round")
        ax.scatter(*chain[1:].T, color="#1d3d66", s=28, depthshade=False)
        # gripper: body from the flange to the finger base, two fingers along tool x
        x_t, z_t, tip = T_tool[:3, 0], T_tool[:3, 2], T_tool[:3, 3]
        base, gap = tip - FINGER_LEN * z_t, f["gap"]
        ax.plot(*np.array([chain[-1], base]).T, color="#444444", lw=5)
        ax.plot(*np.array([base - (gap / 2 + 0.01) * x_t, base + (gap / 2 + 0.01) * x_t]).T, color="#444444", lw=4)
        for side in (-1, 1):
            ax.plot(*np.array([base + side * gap / 2 * x_t, tip + side * gap / 2 * x_t]).T, color="#222222", lw=3)
        ax.scatter(*T_cam[:3, 3], color="#2ca02c", s=40, marker="s", depthshade=False)
        T_obj = f["T_obj"]
        if f["label"].startswith("home: camera"):
            ax.plot(*np.array([T_cam[:3, 3], T_obj[:3, 3] + [0, 0, OBJECT_SIZE[2] / 2]]).T, "g--", lw=1)
        for a, b in box_edges(T_obj):
            ax.plot(*np.array([a, b]).T, color="#e07b00", lw=2)
        ax.plot(*np.array([T_obj[:3, 3], T_obj[:3, 3] + 0.04 * T_obj[:3, 0]]).T, color="#e07b00", lw=1)
        for a, b in box_edges(T_base_place, (OBJECT_SIZE[0], OBJECT_SIZE[1], 0.0)):
            ax.plot(*np.array([a, b]).T, color="#999999", lw=1, ls=":")
        ax.plot(*np.array([T_base_place[:3, 3], T_base_place[:3, 3] + 0.04 * T_base_place[:3, 0]]).T,
                color="#999999", lw=1, ls=":")
        if i > 0:
            ax.plot(*tips[: i + 1].T, color="#9467bd", lw=1)
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(hi - lo))
        ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")
        ax.view_init(elev=22, azim=-35)
        ax.set_title(f"{f['label']}\njoints (servo deg) {np.round(f['servo'], 1) + 0.0}", fontsize=9)

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), init_func=lambda: None,
                                   interval=1000 / fps, repeat=False)
    if show:
        plt.show()
    else:
        writer = animation.FFMpegWriter(fps=fps) if out.endswith(".mp4") else animation.PillowWriter(fps=fps)
        anim.save(out, writer=writer, dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Simulate the arm picking and placing the object.")
    parser.add_argument("--object", type=float, nargs=4, metavar=("X", "Y", "Z", "YAW"),
                        default=[-0.40, 0.10, 0.0, 25.0],
                        help="object pose in the base frame: x y z [m] (bottom centre) and yaw [deg]")
    parser.add_argument("--pose", help="4x4 object pose in the camera frame, image taken at home (.txt or .npy); "
                                       "its origin must be the object's bottom centre with +z up; overrides --object")
    parser.add_argument("--place", type=float, nargs="+", metavar="XY[Z]", default=None,
                        help="where to put it down: x y [z] in the base frame [m]; z defaults to the table "
                             "(default spot: 5 cm / 25 cm from the object)")
    parser.add_argument("--table-z", type=float, default=None,
                        help="table height [m] (default: the plane the object sits on)")
    parser.add_argument("--save", default="pick.gif", help="output .gif or .mp4 (default pick.gif)")
    parser.add_argument("--show", action="store_true", help="open a window instead of saving")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    if not args.show:
        if not args.save.endswith((".gif", ".mp4")):
            parser.error("--save must end in .gif or .mp4")
        if args.save.endswith(".mp4"):
            from matplotlib import animation
            if not animation.writers.is_available("ffmpeg"):
                parser.error("saving .mp4 needs ffmpeg installed; use a .gif instead")
    if args.place is not None and len(args.place) not in (2, 3):
        parser.error("--place takes x y or x y z")

    if args.pose:
        T_base_object = fk(np.zeros(6), "camera") @ load_pose(args.pose)
        if T_base_object[2, 2] < math.cos(math.radians(10)):
            sys.exit("the --pose object's +z axis does not point up (more than 10 deg off); this simulation "
                     "needs the object's frame at its bottom centre with +z up")
    else:
        x, y, z, yaw = args.object
        T_base_object = pose(rot_z(math.radians(yaw)), [x, y, z])
    table_z = args.table_z if args.table_z is not None else float(T_base_object[2, 3])
    if T_base_object[2, 3] < table_z - 1e-9:
        sys.exit(f"the object (z {T_base_object[2, 3]:.3f} m) is below the table (z {table_z:.3f} m)")
    if args.place is None:
        px, py, pz = T_base_object[0, 3] + 0.05, T_base_object[1, 3] + 0.25, table_z
    else:
        px, py = args.place[:2]
        pz = args.place[2] if len(args.place) == 3 else table_z
        if abs(pz - table_z) > 1e-3:
            sys.exit(f"the place spot must be on the table: z {pz:.3f} m vs table {table_z:.3f} m")
    T_base_place = pose(T_base_object[:3, :3], [px, py, pz])

    try:
        frames, table, waypoints, problems = plan(T_base_object, T_base_place, table_z)
    except IKError as e:
        sys.exit(f"cannot plan the pick and place: {e}")

    np.set_printoptions(precision=2, suppress=True)
    print(f"object at {T_base_object[:3, 3]} m, place at {T_base_place[:3, 3]} m, table z {table.z:.3f} m")
    actions = {("pick", "grasp"): "close the gripper", ("place", "place"): "open the gripper"}
    for phase, name, q in waypoints:
        action = actions.get((phase, name))
        print(f"  {phase:<5} {name:<9} servo deg {np.round(q, 2) + 0.0}" + (f"  -> {action}" if action else ""))
    if problems:
        print("SAFETY CHECK FAILED:\n  " + "\n  ".join(dict.fromkeys(problems)))
        sys.exit(1)
    print(f"safety check passed: every move checked at {CHECK_DEG:g} deg steps, arm and carried object above "
          f"the table, object put down exactly at the place spot")
    print(f"rendering {len(frames)} frames ...")
    animate(frames, table, T_base_place, args.save, args.show, args.fps)
    if not args.show:
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
