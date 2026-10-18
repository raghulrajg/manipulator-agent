"""
manipulator_kinematics.py
--------------------------
Forward and inverse kinematics of the 6-DOF arm on each mobile manipulator,
built from your three standard-DH tables. dh_transform(),
forward_kinematics_6dof() and JointMap are unchanged from
effective_transformation.py / joint_mapping.py; the tables are:

    "flange"   no end effector    (effective_transformation.py)
    "tool"     gripper tool tip   (tooltip_transformation.py)
    "camera"   wrist camera       (camera_tranformation.py)

Conventions: poses are 4x4 homogeneous matrices in the arm's base frame
(frame 0 of the tables), in meters. Joint vectors called q are the "visual"
angles of joint_mapping.py in radians (all zero at home); anything named
*_servo_deg is what the controller reports / accepts, in degrees.

Tool and camera poses are flange FK times a fixed offset, read off their
tables once at home (T_FLANGE_TOOL / T_FLANGE_CAMERA); keep editing the
tables below and the offsets follow them. Your updated tool table (d5 0.12,
d6 0.285, no joint 6 offset) is a plain extension along the flange axis --
the tip 0.197 m out from the flange -- so T_FLANGE_TOOL is exact for every
joint angle. The camera table still carries a 76.5 mm sideways offset in d5
(0.1965 instead of 0.12). A d5 offset shifts joint 6's axis instead of
turning with the flange (the table drifts by 2 * 76.5 mm * sin(q6 / 2),
108 mm at q6 = 90 deg), which is why the camera uses the fixed offset read
at q6 = 0 rather than its table directly.

Three theta offsets differ from your scripts: joint 1 (+180 deg), joint 3
(-90 instead of +90) and joint 5 (-90 instead of +90). With your offsets the
model's home pose is the mirror twin of the real one -- the same side view,
but the shoulder offset d4 on the other side (235 mm off) and joints 2-4
turning the opposite way from home. The corrected chain matches Dobot's
official Nova 2 model (dobot_description/urdf/nova2_robot.urdf in
github.com/Dobot-Arm/TCP-IP-ROS-6AXis, whose driver feeds the controller's
joint angles straight in) within 0.02 mm, while home, the flange
orientation there and the tool / camera offsets stay exactly as before.
Check on the robot: at home, with no tool set, the pendant should read the
flange at about X -345, Y +117.5, Z 415.4 mm.

Inverse kinematics is closed form (ik_flange): up to 8 solutions, one per
shoulder x wrist x elbow branch. ik_solutions() turns them into servo angles
within the joint limits, drops every one that puts any part of the arm
below the table (TableConstraint), and orders the rest nearest the current
joint state first.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def dh_transform(alpha, a, d, theta):
    """
    Computes the 4x4 Homogeneous Transformation Matrix for a single joint
    given standard Denavit-Hartenberg parameters:
    - alpha : link twist (radians)
    - a     : link length (meters)
    - d     : link offset (meters)
    - theta : joint angle (radians)
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1],
    ])
    return T


def forward_kinematics_6dof(dh_table):
    """
    dh_table: Nx4 numpy array where each row is [alpha, a, d, theta]

    Returns:
      T_effective : T_0^N (base to end-effector)
      transforms  : list of individual transforms T_{i-1}^{i}
      cumulative  : list of cumulative transforms T_0^{i}
    """
    T_effective = np.eye(4)
    transforms = []
    cumulative = []

    for row in dh_table:
        alpha, a, d, theta = row
        T_i = dh_transform(alpha, a, d, theta)
        T_effective = T_effective @ T_i
        transforms.append(T_i)
        cumulative.append(T_effective.copy())

    return T_effective, transforms, cumulative


class JointMap:
    """
    Maps real servo angles (what the Dobot reports) <-> visual/model angles
    (what goes into the DH table).

        visual = direction * (servo - home_servo)
        servo  = home_servo + direction * visual

    At the home pose every visual angle is 0.
    direction[i] = +1 -> same rotation sense as the servo
    direction[i] = -1 -> flipped
    """

    def __init__(self, home_servo_deg, direction=(1, 1, 1, 1, 1, 1)):
        self.home_servo_deg = np.asarray(home_servo_deg, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        assert self.home_servo_deg.shape == (6,)
        assert self.direction.shape == (6,)
        assert np.all(np.abs(self.direction) == 1), "direction must be +1 or -1"

    def servo_to_visual(self, servo_deg):
        """servo angles [deg] -> visual angles [rad] for the DH table"""
        servo = np.asarray(servo_deg, dtype=float)
        return np.radians(self.direction * (servo - self.home_servo_deg))

    def visual_to_servo(self, visual_rad):
        """visual angles [rad] -> servo angles [deg] to command the robot"""
        visual = np.asarray(visual_rad, dtype=float)
        return self.home_servo_deg + self.direction * np.degrees(visual)


# ----------------------------------------------------------------------
# EDIT HERE
# ----------------------------------------------------------------------
joint_map = JointMap(
    home_servo_deg=[-180, 0, -90, 0, 90, 90],   # j1..j6 at your home pose
    direction=[1, 1, 1, 1, 1, 1],               # set any to -1 to flip that joint
)

# Servo limits [deg], j1..j6, that every IK solution must respect: the
# joint limits of Dobot's official Nova 2 URDF. Tighten them to your
# controller's soft limits if those are narrower.
SERVO_LIMITS_DEG = np.array([
    [-359.8, 359.8],
    [-150.0, 150.0],
    [-150.0, 150.0],
    [-359.8, 359.8],
    [-359.8, 359.8],
    [-359.8, 359.8],
])

# Standard DH rows [alpha (rad), a (m), d (m), theta offset (rad)], with
# theta = visual angle + offset. Your tables, except the joint 1, 3 and 5
# offsets (see the top of this file; yours were 0, +pi/2 and +pi/2).
# Joints 1-4 are the same in all three.
ARM_DH = [
    [np.pi / 2, 0.0,   0.2234, np.pi],       # Joint 1
    [0.0,       0.28,  0.0,    np.pi / 2],   # Joint 2
    [0.0,       0.225, 0.0,    -np.pi / 2],  # Joint 3
    [np.pi / 2, 0.0,   0.1175, np.pi / 2],   # Joint 4
]
WRIST_DH = {                                  # Joints 5 and 6
    "flange": [[np.pi / 2, 0.0, 0.12,   -np.pi / 2],
               [0.0,       0.0, 0.088,  0.0]],
    "tool":   [[np.pi / 2, 0.0, 0.12,   -np.pi / 2],     # updated tooltip table
               [0.0,       0.0, 0.285,  0.0]],
    "camera": [[np.pi / 2, 0.0, 0.1965, -np.pi / 2],
               [0.0,       0.0, 0.138,  np.pi / 2]],
}
# ----------------------------------------------------------------------


def build_dh_params(q, table="flange"):
    """
    q: 6 visual angles in radians (output of joint_map.servo_to_visual).
    table: "flange", "tool" or "camera".
    Rows are [alpha, a, d, theta], same layout as your original tables.
    """
    rows = np.array(ARM_DH + WRIST_DH[table], dtype=float)
    rows[:, 3] += np.asarray(q, dtype=float)
    return rows


def _flange_to(table):
    """Fixed flange -> end transform, read off `table` and the flange table
    at home. It doesn't depend on joints 1-5, so flange FK times this equals
    `table` exactly whenever q6 = 0."""
    home = np.zeros(6)
    T_base_flange = forward_kinematics_6dof(build_dh_params(home, "flange"))[0]
    T_base_end = forward_kinematics_6dof(build_dh_params(home, table))[0]
    return np.linalg.inv(T_base_flange) @ T_base_end


T_FLANGE_TOOL = _flange_to("tool")
T_FLANGE_CAMERA = _flange_to("camera")
_FLANGE_TO = {"flange": np.eye(4), "tool": T_FLANGE_TOOL, "camera": T_FLANGE_CAMERA}


def fk(q, frame="tool"):
    """Pose of "flange", "tool" (tip) or "camera" in the base frame at
    visual angles q [rad]."""
    T_base_flange = forward_kinematics_6dof(build_dh_params(q, "flange"))[0]
    return T_base_flange @ _FLANGE_TO[frame]


def pose(R, p):
    """4x4 homogeneous transform from a 3x3 rotation and a position."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---- inverse kinematics ----
# Flange-table geometry the closed form below relies on: alpha = [90, 0, 0,
# 90, 90, 0] deg (joints 2-4 parallel) and a1 = a4 = a5 = a6 = 0.
_D1, _A2, _A3, _D4 = ARM_DH[0][2], ARM_DH[1][1], ARM_DH[2][1], ARM_DH[3][2]
_D5, _D6 = WRIST_DH["flange"][0][2], WRIST_DH["flange"][1][2]
_THETA_OFFSET = np.array([row[3] for row in ARM_DH + WRIST_DH["flange"]])


def _wrap(angles):
    return (np.asarray(angles) + np.pi) % (2 * np.pi) - np.pi


def _singular_th6(p5, x6, y6, th6_ref, spread=16):
    """Joint 6 DH angles to try at the wrist singularity, where joint 6 is
    free: th6_ref if it puts frame 4 within the two links' reach (else the
    nearest angle that does), plus `spread` angles across each side of the
    reachable range, so joint limits on 2-4 can't rule them all out. Frame
    4's origin is p5 - d5 * (sin th6 x6 + cos th6 y6), so its squared
    in-plane distance from the shoulder is L - K cos(th6 - beta)."""
    v = p5 - np.array([0.0, 0.0, _D1])
    vx, vy = float(v @ x6), float(v @ y6)
    K = 2.0 * _D5 * math.hypot(vx, vy)
    if K < 1e-12:
        return [th6_ref]  # frame 4 doesn't move with joint 6
    L = float(v @ v) + _D5 * _D5 - _D4 * _D4
    beta = math.atan2(vx, vy)
    # reach (a2 - a3)^2 <= L - K cos(th6 - beta) <= (a2 + a3)^2
    near = math.acos(min(1.0, max(-1.0, (L - (_A2 - _A3) ** 2) / K)))  # |th6 - beta| >= near
    far = math.acos(min(1.0, max(-1.0, (L - (_A2 + _A3) ** 2) / K)))   # |th6 - beta| <= far
    # 1e-6 rad inside the interval: right at its edge the elbow is fully
    # stretched and the reach test can miss by a rounding error
    margin = min(1e-6, max(0.0, (far - near) / 2))
    near, far = near + margin, far - margin
    d = (th6_ref - beta + math.pi) % (2 * math.pi) - math.pi
    angles = [beta + math.copysign(min(max(abs(d), near), far), d)]
    if far > near:
        angles += [beta + side * x for side in (1.0, -1.0) for x in np.linspace(near, far, spread)]
    return angles


def ik_flange(T_base_flange, q6_ref=0.0):
    """
    Every closed-form solution q (visual angles, rad, wrapped to [-pi, pi))
    of fk(q, "flange") == T_base_flange -- up to 8, one per shoulder (j1) x
    wrist (j5) x elbow (j3) branch; at the wrist singularity, where joint 6
    is free, a spread of them instead (see below). Empty list if the pose is
    out of reach.

    With th = q + theta offset, p5 = flange - d6 * z6 (frame 5's origin)
    and z1 = (sin th1, -cos th1, 0) (the parallel axis of joints 2-4):
      th1  p5 sits d4 off the arm plane:            p5 . z1 = d4
      th5  z6 . z1 = -cos th5
      th6  (z1 . x6, z1 . y6) = sin th5 * (cos th6, -sin th6)
      th2, th3  planar 2-link arm (a2, a3) to frame 4's origin in frame 1
      th4  th2 + th3 + th4 = direction of x4 in frame 1
    th5 comes from atan2(|sin th5|, cos th5) rather than acos, which keeps
    it accurate right up to the wrist singularity (|sin th5| < 1e-8, the
    threshold of the UR reference IK). There joints 2, 3, 4 and 6 are all
    parallel and the pose doesn't fix joint 6: it is tried at q6_ref (or the
    nearest angle that puts frame 4 within the two links' reach) and across
    the reachable range (_singular_th6); th5 is fitted to each and joints
    2-4 follow.
    """
    T = np.asarray(T_base_flange, dtype=float)
    x6, y6, z6 = T[:3, 0], T[:3, 1], T[:3, 2]
    p5 = T[:3, 3] - _D6 * z6

    r = math.hypot(p5[0], p5[1])
    if r < _D4 - 1e-12:  # tolerance: at the shoulder singularity r = d4 can round 1 ulp low
        return []
    phi = math.atan2(p5[1], p5[0])
    s = math.asin(min(1.0, _D4 / r))

    solutions = []
    for th1 in (phi + s, phi + math.pi - s):
        z1 = np.array([math.sin(th1), -math.cos(th1), 0.0])
        c5 = -float(z6 @ z1)
        a, b = float(z1 @ x6), float(z1 @ y6)  # = sin th5 * (cos th6, -sin th6)
        s5 = math.hypot(a, b)                   # |sin th5|
        if s5 < 1e-8:  # wrist singularity: joint 6 is free (see above)
            wrists = [(math.atan2(a * math.cos(th6) - b * math.sin(th6), c5), th6)
                      for th6 in _singular_th6(p5, x6, y6, q6_ref + _THETA_OFFSET[5])]
        else:
            wrists = [(math.atan2(sign * s5, c5), math.atan2(-sign * b, sign * a))
                      for sign in (1.0, -1.0)]
        for th5, th6 in wrists:
            T_1_4 = (np.linalg.inv(dh_transform(np.pi / 2, 0.0, _D1, th1)) @ T
                     @ np.linalg.inv(dh_transform(0.0, 0.0, _D6, th6))
                     @ np.linalg.inv(dh_transform(np.pi / 2, 0.0, _D5, th5)))
            px, py = T_1_4[0, 3], T_1_4[1, 3]
            c3 = (px * px + py * py - _A2 * _A2 - _A3 * _A3) / (2 * _A2 * _A3)
            if abs(c3) > 1.0 + 1e-9:
                continue  # frame 4's origin out of the 2-link arm's reach on this branch
            c3 = float(np.clip(c3, -1.0, 1.0))
            for th3 in (math.acos(c3), -math.acos(c3)):
                th2 = math.atan2(py, px) - math.atan2(_A3 * math.sin(th3), _A2 + _A3 * math.cos(th3))
                th4 = math.atan2(T_1_4[1, 0], T_1_4[0, 0]) - th2 - th3
                theta = np.array([th1, th2, th3, th4, th5, th6])
                solutions.append(_wrap(theta - _THETA_OFFSET))
    return solutions


def to_servo_near(q, ref_servo_deg, limits=SERVO_LIMITS_DEG):
    """
    Visual angles [rad] -> servo angles [deg], taking for each joint the
    360-degree equivalent nearest ref_servo_deg that lies inside `limits`,
    so joints with a +-360 range never unwind a full turn for nothing.
    None if some joint has no equivalent inside its limits. Angles within
    1e-9 deg of a limit count as on it (a joint sitting exactly at its
    limit comes back from the IK a rounding error outside).
    """
    servo = joint_map.visual_to_servo(q)
    out = np.empty(6)
    for i in range(6):
        lo, hi = limits[i]
        options = servo[i] + 360.0 * np.arange(-2, 3)
        options = options[(options >= lo - 1e-9) & (options <= hi + 1e-9)]
        if options.size == 0:
            return None
        out[i] = min(max(options[np.argmin(np.abs(options - ref_servo_deg[i]))], lo), hi)
    return out


@dataclass
class TableConstraint:
    """
    The table the object lies on: a horizontal surface at height z in the
    arm's base frame. Nothing may go below it to reach the object -- elbow,
    wrist, flange and camera stay link_clearance above it, the gripper
    (flange to tool tip) tip_clearance above it, since the tip has to get
    down to the object.

    x_range / y_range bound the table's footprint in the base frame. Leave
    them None to treat the surface as an endless floor at that height (the
    strictest choice); set them if the arm also works beside the table,
    lower than its top.
    """
    z: float
    link_clearance: float = 0.05
    tip_clearance: float = 0.005
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None

    def _over_table(self, p):
        return all(r is None or r[0] <= v <= r[1]
                   for v, r in ((p[0], self.x_range), (p[1], self.y_range)))

    def violations(self, q) -> List[str]:
        """Parts of the arm that are too low at visual angles q, e.g.
        ["forearm 3.2 cm too low"]. Empty list = clear of the table."""
        _, _, cumulative = forward_kinematics_6dof(build_dh_params(q, "flange"))
        o = [T[:3, 3] for T in cumulative]  # origins of frames 1..6 (shoulder .. flange)
        tip = (cumulative[-1] @ T_FLANGE_TOOL)[:3, 3]
        camera = (cumulative[-1] @ T_FLANGE_CAMERA)[:3, 3]
        links = (
            ("upper arm", o[0], o[1], self.link_clearance),
            ("forearm",   o[1], o[2], self.link_clearance),
            ("wrist",     o[2], o[3], self.link_clearance),
            ("wrist",     o[3], o[4], self.link_clearance),
            ("wrist",     o[4], o[5], self.link_clearance),
            ("camera",    o[5], camera, self.link_clearance),
            ("gripper",   o[5], tip, self.tip_clearance),
        )
        too_low = {}
        for name, start, end, clearance in links:
            for t in np.linspace(0.0, 1.0, 5):
                p = start + t * (end - start)
                short = self.z + clearance - p[2]
                if short > 1e-9 and self._over_table(p):
                    too_low[name] = max(too_low.get(name, 0.0), short)
        return [f"{name} {short * 100:.1f} cm too low" for name, short in too_low.items()]

    def path_violations(self, q_from, q_to, steps=20) -> List[str]:
        """The same check along the straight joint-space move q_from -> q_to
        (visual angles), i.e. the path a joint move such as Dobot MovJ takes.
        Empty list = clear; otherwise the violations at the first bad step."""
        q_from, q_to = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
        for t in np.linspace(0.0, 1.0, steps + 1)[1:]:
            bad = self.violations(q_from + t * (q_to - q_from))
            if bad:
                return [f"{v} at {t:.0%} of the move" for v in bad]
        return []


class IKError(Exception):
    """No IK solution reaches the pose within the joint limits while keeping
    the arm above the table; the message says what ruled each one out."""
    pass


def ik_solutions(T_base_tool, ref_servo_deg, table: Optional[TableConstraint] = None,
                 limits=SERVO_LIMITS_DEG) -> List[np.ndarray]:
    """
    Every usable IK solution for the tool tip pose T_base_tool, as servo
    angles [deg], nearest ref_servo_deg (normally the current joint state)
    first. Usable = inside the joint limits and, if a table is given, with
    no part of the arm below it. Raises IKError if none is usable.
    """
    ref = np.asarray(ref_servo_deg, dtype=float)
    T_base_tool = np.asarray(T_base_tool, dtype=float)
    tip = np.round(T_base_tool[:3, 3], 4).tolist()
    T_base_flange = T_base_tool @ np.linalg.inv(T_FLANGE_TOOL)

    candidates = ik_flange(T_base_flange, q6_ref=joint_map.servo_to_visual(ref)[5])
    if not candidates:
        raise IKError(f"tool tip pose at {tip} is out of the arm's reach")

    usable, rejected = [], []
    for q in candidates:
        servo = to_servo_near(q, ref, limits)
        if servo is None:
            rejected.append("outside joint limits")
            continue
        if any(np.abs(servo - u).max() < 1e-6 for u in usable):
            continue  # duplicate branch (singular pose)
        bad = table.violations(joint_map.servo_to_visual(servo)) if table else []
        if bad:
            rejected.append("below the table: " + ", ".join(bad))
            continue
        usable.append(servo)

    if not usable:
        raise IKError(f"none of the {len(candidates)} IK solutions for the tool tip at {tip} "
                      f"is usable: " + "; ".join(dict.fromkeys(rejected)))
    return sorted(usable, key=lambda s: float(np.linalg.norm(s - ref)))


def solve_ik(T_base_tool, ref_servo_deg, table: Optional[TableConstraint] = None,
             limits=SERVO_LIMITS_DEG) -> np.ndarray:
    """The usable tool-tip IK solution nearest ref_servo_deg, in servo
    degrees (see ik_solutions)."""
    return ik_solutions(T_base_tool, ref_servo_deg, table, limits)[0]
