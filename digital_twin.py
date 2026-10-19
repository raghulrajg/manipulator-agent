"""
digital_twin.py
----------------
Digital twin of the Nova 2: a browser app (twin_web/) showing the robot
with Dobot's own Nova 2 model (URDF + meshes), moving exactly as the real
arm moves -- and from which the arm can be moved and the pick and place run.

    robot -> twin   the controller's real-time feedback (port 30004, actual
                    joints every 8 ms) is pushed to every browser 30 times a
                    second, whatever moved the arm: this app, the pendant,
                    a script, or a hand in drag mode
    twin -> robot   browser commands are checked here (armed, joint limits,
                    whole path above the table, speed cap) and sent as
                    MovJ / MovL on port 29999; the gripper over RS485.
                    What the twin then shows is again the robot's feedback.

With --sim the "robot" is fake_dobot.py (same TCP commands, same feedback
port, Nova 2 speeds), so the app is also the simulator, on the same code
path as the real arm.

Run:
    python3 digital_twin.py --sim                               # then open http://localhost:8080
    python3 digital_twin.py --robot 192.168.5.1 --table-z -0.05 # real arm, real gripper (left, ID 21)
    python3 digital_twin.py --robot 192.168.5.1 --table-z -0.05 --mqtt   # + "Detect" with the vision PC
"""

import argparse
import asyncio
import json
import math
import queue
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np

from dobot_feedback import FeedbackReader
from dobot_tcp import MODE_IDLE, MODE_NAMES, DobotError, DobotTCP
from gripper import LEFT, RIGHT, FakeGripper, GripperError
from manipulator_kinematics import (
    SERVO_LIMITS_DEG, T_FLANGE_CAMERA, T_FLANGE_TOOL, IKError, TableConstraint, fk, ik_solutions,
    joint_map, pose, rot_z,
)
from mqtt_link import LocalBus, MessageError
from pick_runner import DEMO_T_CAMERA_OBJECT, PickFailed, PickRunner, start_fake_vision

WEB = Path(__file__).resolve().parent / "twin_web"
URDF = WEB / "model" / "nova2" / "nova2_robot.urdf"
T_LINK6_FLANGE = pose(rot_z(math.pi / 2), [0.0, 0.0, 0.0])   # Dobot's link6 -> the DH flange frame
HOME = joint_map.home_servo_deg
LINE_STEP = 0.005        # [m] straight-line moves are checked every 5 mm
LINE_MAX_JUMP = 10.0     # [deg] largest joint change per step on a line (more = IK branch switch)
FEEDBACK_STALE = 0.5     # [s] older feedback = robot not connected


class TwinError(RuntimeError):
    pass


# ============================================================
# ROBOT MODEL (Dobot's URDF)
# ============================================================

def _floats(text, n=3):
    return [float(v) for v in (text or " ".join(["0"] * n)).split()]


def _origin(xyz, rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])   # URDF rpy: Rz(yaw) Ry(pitch) Rx(roll)
    return pose(R, xyz)


def load_model(path=URDF):
    """URDF -> what the browser needs: each link's mesh and colour, each
    joint's fixed origin (4x4, row-major) and limits, plus where the
    gripper tip, the camera and the DH flange sit on link 6."""
    root = ET.parse(path).getroot()
    links = {}
    for link in root.findall("link"):
        mesh = link.find("visual/geometry/mesh")
        color = link.find("visual/material/color")
        links[link.get("name")] = {
            "mesh": f"web/model/nova2/{mesh.get('filename')}" if mesh is not None else None,
            "color": _floats(color.get("rgba"), 4) if color is not None else [0.8, 0.8, 0.9, 1.0],
        }
    joints = []
    for j in root.findall("joint"):
        if j.get("type") != "revolute":
            continue
        o, lim = j.find("origin"), j.find("limit")
        joints.append({
            "name": j.get("name"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "origin": _origin(_floats(o.get("xyz")), _floats(o.get("rpy"))).round(8).tolist(),
            "axis": _floats(j.find("axis").get("xyz")),
            "limit_deg": [math.degrees(float(lim.get("lower"))), math.degrees(float(lim.get("upper")))],
        })
    return {
        "base": "base_link", "links": links, "joints": joints,
        "link6_flange": T_LINK6_FLANGE.tolist(),
        "link6_tool": (T_LINK6_FLANGE @ T_FLANGE_TOOL).round(8).tolist(),
        "link6_camera": (T_LINK6_FLANGE @ T_FLANGE_CAMERA).round(8).tolist(),
        "servo_limits_deg": SERVO_LIMITS_DEG.tolist(),
        "home": HOME.tolist(),
    }


def model_fk(model, servo_deg):
    """Link 6 pose from the URDF chain at these controller angles (the
    browser does the same; the tests compare it with manipulator_kinematics)."""
    T = np.eye(4)
    for j, angle in zip(model["joints"], servo_deg):
        T = T @ np.array(j["origin"]) @ pose(rot_z(math.radians(angle)), [0, 0, 0])
    return T


# ============================================================
# GRIPPER AS THE TWIN SEES IT
# ============================================================

class TwinGripper:
    """Wraps the real (or fake) gripper so the twin knows its state and
    can carry the object. In simulation the fingers only close on the
    object if the tool tip is really at it."""

    def __init__(self, inner, twin, sim):
        self.inner, self.twin, self.sim = inner, twin, sim

    def open(self, speed=100, wait=True):
        self.inner.open(speed, wait)
        self.twin._gripper_changed(0, False)

    def close(self, speed=50):
        if self.sim and isinstance(self.inner, FakeGripper):
            self.inner.object_position = 140 if self.twin._object_between_fingers() else None
        result = self.inner.close(speed)
        self.twin._gripper_changed(result.position, result.holding)
        return result

    def disconnect(self):
        self.inner.disconnect()


# ============================================================
# THE TWIN
# ============================================================

class Twin:
    def __init__(self, robot, feedback, gripper, link=None, table_z=0.0, home=None, max_speed=50,
                 place_xy=None, sim=False, object_diameter=0.06, log=print):
        self.robot, self.feedback = robot, feedback
        self.sim, self.table_z = sim, table_z
        self.home = np.asarray(HOME if home is None else home, dtype=float)
        self.max_speed = int(max_speed)
        self.speed = min(20, self.max_speed)
        self.lock = threading.RLock()
        self.armed = False
        self.task = self.step = None
        self.cancel = threading.Event()
        self.model_error_mm = None
        self.object = None          # {"T": 4x4, "diameter", "attached", "T_tool_obj", "source"}
        self.object_diameter = object_diameter
        self.place = None if place_xy is None else [float(v) for v in place_xy]
        self.ghost = None           # joints shown as a see-through robot
        self.plan = None            # {"moves": [...], "path": [[x, y, z], ...]}
        self.grip = {"position": 0, "holding": False}
        self._log = deque(maxlen=300)
        self._log_seq = 0
        self._print = log
        self.link = link or LocalBus()
        self.gripper = TwinGripper(gripper, self, sim)
        self.runner = PickRunner(robot, self.gripper, self.link, home=self.home, table_z=table_z,
                                 log=lambda text: self.log(text, "info"))
        self.runner.on_status = lambda state, detail: self._set_step(f"{state}" + (f": {detail}" if detail else ""))
        self.fake = None            # the simulated controller, with --sim
        self._tasks = queue.Queue()
        threading.Thread(target=self._worker, daemon=True, name="twin-tasks").start()

    # ------------------------------------------------------------ log / state
    def log(self, text, level="info"):
        with self.lock:
            self._log_seq += 1
            self._log.append({"seq": self._log_seq, "t": time.time(), "level": level, "text": str(text)})
        self._print(f"[{level}] {text}")

    def logs_after(self, seq):
        with self.lock:
            return [e for e in self._log if e["seq"] > seq]

    def _set_step(self, text):
        with self.lock:
            self.step = text

    def robot_state(self):
        s = self.feedback.latest()
        age = self.feedback.age()
        if s is None:
            return {"connected": False, "age_ms": None}
        q = s["q_actual"]
        T_tool = fk(joint_map.servo_to_visual(q), "tool")
        return {
            "connected": age < FEEDBACK_STALE, "age_ms": round(age * 1000, 1),
            "mode": int(s["robot_mode"]), "mode_name": MODE_NAMES.get(int(s["robot_mode"]), "?"),
            "joints": np.round(q, 4).tolist(), "tcp": np.round(s["tool_vector_actual"], 3).tolist(),
            "drag": int(s["drag_status"]), "queue": int(s["current_command_id"]),
            "speed_scaling": round(float(s["speed_scaling"]), 3), "robot_type": int(s["robot_type"]),
            "tool": T_tool.round(6).tolist(),
        }

    def snapshot(self):
        robot = self.robot_state()
        with self.lock:
            obj = None
            if self.object is not None:
                T = self.object["T"]
                if self.object["attached"] and robot.get("tool") is not None:
                    T = np.array(robot["tool"]) @ self.object["T_tool_obj"]
                    self.object["T"] = T
                obj = {"T": T.round(6).tolist(), "diameter": self.object["diameter"],
                       "attached": self.object["attached"], "source": self.object["source"]}
            return {
                "type": "state", "t": time.time(), "robot": robot, "armed": self.armed,
                "task": self.task, "step": self.step, "speed": self.speed, "max_speed": self.max_speed,
                "gripper": dict(self.grip), "object": obj, "place": self.place, "table_z": self.table_z,
                "ghost": None if self.ghost is None else np.round(self.ghost, 4).tolist(),
                "plan": self.plan, "model_error_mm": self.model_error_mm, "sim": self.sim,
                "log_seq": self._log_seq,
            }

    # ------------------------------------------------------------ checks
    def _joints_now(self):
        """Where the arm is, for planning a command: the controller's
        GetAngle (the feedback can be a packet behind right after a move)."""
        if self.feedback.latest() is None or self.feedback.age() > FEEDBACK_STALE:
            raise TwinError("no feedback from the robot")
        return self.robot.get_angle()

    def _table(self):
        return None if self.table_z is None else TableConstraint(self.table_z)

    def check_joint_move(self, q_from, q_to):
        """Joint limits and the joint-space path (MovJ) above the table."""
        q_to = np.asarray(q_to, dtype=float)
        if q_to.shape != (6,) or not np.all(np.isfinite(q_to)):
            raise TwinError("need 6 finite joint angles")
        lo, hi = SERVO_LIMITS_DEG[:, 0], SERVO_LIMITS_DEG[:, 1]
        bad = np.where((q_to < lo) | (q_to > hi))[0]
        if bad.size:
            raise TwinError("outside the joint limits: " + ", ".join(
                f"J{i + 1} {q_to[i]:.1f} (limits {lo[i]:.0f}..{hi[i]:.0f})" for i in bad))
        table = self._table()
        if table is not None:
            bad = table.path_violations(joint_map.servo_to_visual(q_from), joint_map.servo_to_visual(q_to))
            if bad:
                raise TwinError("the move goes below the table: " + ", ".join(bad))
        return q_to

    def check_line(self, q_from, T_to):
        """A straight tool line (MovL) from q_from to T_to: IK every 5 mm on
        one branch, above the table. Returns the joints at the end."""
        T_from = fk(joint_map.servo_to_visual(q_from), "tool")
        if not np.allclose(T_from[:3, :3], T_to[:3, :3], atol=1e-6):
            raise TwinError("a straight-line move here keeps the tool's orientation; use a joint move to turn it")
        n = max(1, math.ceil(np.linalg.norm(T_to[:3, 3] - T_from[:3, 3]) / LINE_STEP))
        q, table = np.asarray(q_from, dtype=float), self._table()
        for k in range(1, n + 1):
            T = T_to.copy()
            T[:3, 3] = T_from[:3, 3] + (T_to[:3, 3] - T_from[:3, 3]) * k / n
            try:
                q_next = ik_solutions(T, q, table)[0]
            except IKError as e:
                raise TwinError(f"the line cannot be followed at {np.round(T[:3, 3], 3).tolist()}: {e}") from None
            jump = float(np.abs(q_next - q).max())
            if jump > LINE_MAX_JUMP:
                raise TwinError(f"the line passes a singularity ({jump:.0f} deg joint jump); use a joint move")
            if table is not None:
                bad = table.path_violations(joint_map.servo_to_visual(q), joint_map.servo_to_visual(q_next), steps=4)
                if bad:
                    raise TwinError("the line goes below the table: " + ", ".join(bad))
            q = q_next
        return q

    def ik(self, T_tool, ref):
        try:
            return ik_solutions(np.asarray(T_tool, dtype=float), ref, self._table())[0]
        except IKError as e:
            raise TwinError(str(e)) from None

    def _object_between_fingers(self):
        with self.lock:
            if self.object is None or self.object["attached"]:
                return False
            center, d = self.object["T"][:3, 3], self.object["diameter"]
        tip = fk(joint_map.servo_to_visual(self._joints_now()), "tool")[:3, 3]
        return float(np.linalg.norm(tip - center)) <= max(0.012, 0.35 * d)

    def _gripper_changed(self, position, holding):
        q = self._joints_now()
        T_tool = fk(joint_map.servo_to_visual(q), "tool")
        with self.lock:
            self.grip = {"position": int(position), "holding": bool(holding)}
            if holding:
                if self.object is None or self.object["attached"] or \
                        np.linalg.norm(self.object["T"][:3, 3] - T_tool[:3, 3]) > max(0.03, self.object["diameter"]):
                    # something the twin did not know about: show it in the fingers
                    self.object = {"T": T_tool.copy(), "diameter": self.object_diameter, "source": "gripped"}
                self.object["attached"] = True
                self.object["T_tool_obj"] = np.linalg.inv(T_tool) @ self.object["T"]
            elif self.object is not None and self.object.get("attached"):
                self.object["attached"] = False
                T = T_tool @ self.object["T_tool_obj"]   # where the fingers let it go
                if self.table_z is not None:            # ... and it ends up on the table
                    T[2, 3] = self.table_z + self.object["diameter"] / 2
                self.object["T"] = T

    # ------------------------------------------------------------ tasks
    def _queue(self, name, fn):
        with self.lock:
            if self.task is not None:
                raise TwinError(f"busy: {self.task}")
            if not self.armed:
                raise TwinError("not armed: switch on 'Arm' to let the app move the robot")
            self.task, self.step = name, None
            self.cancel.clear()
        self._tasks.put((name, fn))
        return {"queued": name}

    def _worker(self):
        while True:
            name, fn = self._tasks.get()
            self.log(f"{name}: started")
            try:
                fn()
                self.log(f"{name}: done")
            except (PickFailed, IKError, TwinError) as e:
                self.log(f"{name}: {e}", "warn")
            except (DobotError, GripperError, OSError) as e:
                with self.lock:
                    self.armed = False
                self.log(f"{name}: {e} -- disarmed", "error")
            except Exception as e:   # a bug must not kill the worker
                with self.lock:
                    self.armed = False
                self.log(f"{name}: unexpected {type(e).__name__}: {e} -- disarmed", "error")
            finally:
                with self.lock:
                    self.task = None
                    self.ghost = None

    def _speeds(self):
        self.runner.speed = self.speed
        self.runner.line_speed = max(1, self.speed // 2)

    def _move_joints(self, q, what):
        self._speeds()
        self.robot.move_j(q, v=self.runner.speed, a=self.runner.speed, what=what)

    def _move_line(self, q, T, what):
        self._speeds()
        reached = self.runner._reached(T)
        self.robot.move_l(q, reached, v=self.runner.line_speed, a=self.runner.line_speed, what=what)

    # ------------------------------------------------------------ pick & place
    def _object_message(self, q_now):
        with self.lock:
            if self.object is None:
                raise TwinError("no object: click 'Set object' and then the table, or press Detect")
            if self.object["attached"]:
                raise TwinError("the gripper is already holding the object")
            T_base_object = self.object["T"].copy()
            d = self.object["diameter"]
        T_camera_object = np.linalg.inv(fk(joint_map.servo_to_visual(q_now), "camera")) @ T_base_object
        return {"T_camera_object": T_camera_object, "origin": "center", "diameter": d,
                "symmetric": True, "object": "twin", "request_id": None}

    def _plan_view(self, moves, q_start):
        """Moves -> what the browser draws: each waypoint and the tool tip path."""
        path, q = [], np.asarray(q_start, dtype=float)
        for m in moves:
            for s in np.linspace(0.0, 1.0, 12)[1:] if m.kind == "movj" else (1.0,):
                path.append(fk(joint_map.servo_to_visual(q + s * (m.q - q)), "tool")[:3, 3].round(4).tolist())
            q = m.q
        T0 = fk(joint_map.servo_to_visual(q_start), "tool")[:3, 3].round(4).tolist()
        return {"moves": [{"name": m.name, "kind": m.kind, "q": np.round(m.q, 4).tolist(),
                           "xyz": m.T_tool[:3, 3].round(4).tolist(), "then": m.then} for m in moves],
                "path": [T0] + path}

    def plan_pick(self):
        q_now = self._joints_now()
        _, moves, _, _ = self.runner.plan(self._object_message(q_now), q_now, self.place)
        view = self._plan_view(moves, q_now)
        with self.lock:
            self.plan = view
        return view

    # ------------------------------------------------------------ commands
    def handle(self, cmd):
        """One command from a browser -> reply dict. Motions are queued; the
        reply says whether they were accepted, the state shows the rest."""
        try:
            name = cmd.get("cmd")
            if not self.robot.connected:
                # the command port dropped: reconnect (a lost motion is never re-sent)
                self.robot.connect()
                self.log("command connection to the robot re-opened", "warn")
            fn = getattr(self, f"cmd_{name}", None)
            if fn is None:
                raise TwinError(f"unknown command {name!r}")
            return {"ok": True, **(fn(**{k: v for k, v in cmd.items() if k not in ("cmd", "id")}) or {})}
        except (TwinError, IKError, MessageError, DobotError, GripperError, ValueError, TypeError, OSError) as e:
            return {"ok": False, "error": str(e)}

    def cmd_arm(self, on=True):
        if not on:
            with self.lock:
                self.armed = False
            self.log("disarmed")
            return {}
        if not self.robot_state()["connected"]:
            raise TwinError("no feedback from the robot")
        mode = self.robot.robot_mode()   # the controller's own word, not feedback up to 8 ms old
        if mode != MODE_IDLE:
            raise TwinError(f"the robot is {MODE_NAMES.get(mode, mode)}: enable it first")
        if self.table_z is None:
            raise TwinError("no table height (--table-z)")
        if self.model_error_mm is None:
            self.check_model()
        if self.model_error_mm > 3.0:
            raise TwinError(f"the kinematic model is {self.model_error_mm:.1f} mm off the controller: not arming")
        with self.lock:
            self.armed = True
        self.log("armed: the app may move the robot", "warn")
        return {}

    def cmd_stop(self):
        self.cancel.set()
        with self.lock:
            self.armed = False
        try:
            self.robot.stop()
        finally:
            self.log("STOP sent, disarmed", "error")
        return {}

    def cmd_enable(self):
        try:
            self.robot.request_control()
        except DobotError as e:
            self.log(f"RequestControl: {e} (already in TCP mode?)", "warn")
        self.robot.enable()
        t0 = time.monotonic()   # until the feedback (and so every browser) shows it too
        while (self.feedback.latest() or {}).get("robot_mode") != MODE_IDLE and time.monotonic() - t0 < 1.0:
            time.sleep(0.01)
        self.check_model()
        self.log("robot enabled")
        return {}

    def cmd_disable(self):
        with self.lock:
            if self.task:
                raise TwinError(f"busy: {self.task}")
            self.armed = False
        self.robot.disable()
        self.log("robot disabled")
        return {}

    def cmd_clear_error(self):
        self.robot.clear_error()
        self.log("alarms cleared")
        return {}

    def cmd_drag(self, on=True):
        with self.lock:
            if self.task:
                raise TwinError(f"busy: {self.task}")
            self.armed = False   # nothing from the app moves the arm while a hand does
        if on:
            self.robot.start_drag()
            self.log("drag mode: move the arm by hand, the twin follows", "warn")
        else:
            self.robot.stop_drag()
            self.log("drag mode off")
        return {}

    def cmd_speed(self, value):
        value = int(value)
        if not 1 <= value <= self.max_speed:
            raise TwinError(f"speed must be 1..{self.max_speed} %")
        with self.lock:
            self.speed = value
        return {}

    def check_model(self):
        q = self.robot.get_angle()
        measured = self.robot.get_pose()[:3] / 1000
        model = fk(joint_map.servo_to_visual(q), "flange")[:3, 3]
        self.model_error_mm = float(np.linalg.norm(measured - model) * 1000)
        self.log(f"model check: controller vs FK {self.model_error_mm:.2f} mm",
                 "info" if self.model_error_mm <= 3.0 else "error")
        return self.model_error_mm

    # previews (no motion, allowed any time)
    def cmd_preview_joints(self, joints):
        q_to = self.check_joint_move(self._joints_now(), joints)
        with self.lock:
            self.ghost = q_to
        return {"joints": q_to.tolist()}

    def cmd_preview_pose(self, T, linear=False):
        T = _pose_from_client(T)
        q_now = self._joints_now()
        q = self.check_line(q_now, T) if linear else self.check_joint_move(q_now, self.ik(T, q_now))
        with self.lock:
            self.ghost = q
        return {"joints": np.round(q, 4).tolist()}

    def cmd_clear_preview(self):
        with self.lock:
            self.ghost = None
            self.plan = None
        return {}

    # motions (armed, one at a time)
    def cmd_move_joints(self, joints):
        q_to = self.check_joint_move(self._joints_now(), joints)
        with self.lock:
            self.ghost = q_to
        return self._queue("move joints", lambda: self._move_joints(q_to, "move joints"))

    def cmd_jog_joint(self, joint, delta):
        q = self._joints_now()
        i = int(joint)
        if not 0 <= i < 6 or abs(float(delta)) > 30:
            raise TwinError("jog: joint 0..5, at most 30 deg")
        q[i] += float(delta)
        return self.cmd_move_joints(q.tolist())

    def cmd_move_pose(self, T, linear=False):
        T = _pose_from_client(T)
        q_now = self._joints_now()
        if linear:
            q = self.check_line(q_now, T)
            with self.lock:
                self.ghost = q
            return self._queue("move line", lambda: self._move_line(q, T, "move line"))
        q = self.check_joint_move(q_now, self.ik(T, q_now))
        with self.lock:
            self.ghost = q
        return self._queue("move to pose", lambda: self._move_joints(q, "move to pose"))

    def cmd_jog_tool(self, axis, delta):
        """Straight-line jog of the tool tip along a base axis, delta in mm."""
        if axis not in ("x", "y", "z") or abs(float(delta)) > 100:
            raise TwinError("jog: axis x / y / z, at most 100 mm")
        T = fk(joint_map.servo_to_visual(self._joints_now()), "tool")
        T[{"x": 0, "y": 1, "z": 2}[axis], 3] += float(delta) / 1000
        return self.cmd_move_pose(T.tolist(), linear=True)

    def cmd_home(self):
        return self.cmd_move_joints(self.home.tolist())

    def cmd_gripper(self, action):
        if action not in ("open", "close"):
            raise TwinError("gripper: open or close")
        return self._queue(f"gripper {action}", lambda: self.gripper.open() if action == "open" else self.gripper.close())

    # the scene
    def cmd_set_object(self, xyz=None, xy=None, yaw_deg=0.0, diameter=None):
        d = float(diameter or self.object_diameter)
        if not 0.01 <= d <= 0.2:
            raise TwinError("object diameter must be 1..20 cm")
        if xy is not None:
            if self.table_z is None:
                raise TwinError("no table height")
            xyz = [float(xy[0]), float(xy[1]), self.table_z + d / 2]
        xyz = [float(v) for v in xyz]
        with self.lock:
            if self.object is not None and self.object["attached"]:
                raise TwinError("the gripper is holding the object")
            self.object = {"T": pose(rot_z(math.radians(float(yaw_deg))), xyz), "diameter": d,
                           "attached": False, "source": "manual"}
            self.plan = None
        return {}

    def cmd_clear_object(self):
        with self.lock:
            if self.object is not None and self.object["attached"]:
                raise TwinError("the gripper is holding the object")
            self.object = self.plan = None
        return {}

    def cmd_set_place(self, xy=None):
        with self.lock:
            self.place = None if xy is None else [float(xy[0]), float(xy[1])]
            self.plan = None
        return {}

    def cmd_plan(self):
        return {"plan": self.plan_pick()}

    def cmd_pick(self):
        self.plan_pick()   # refuse now if it can't be done

        def run():
            q_now = self._joints_now()
            msg = self._object_message(q_now)
            _, moves, _, _ = self.runner.plan(msg, q_now, self.place)
            with self.lock:
                self.plan = self._plan_view(moves, q_now)
            self._speeds()
            self.runner.execute(moves, self.place)
        return self._queue("pick and place" if self.place else "pick", run)

    def cmd_detect(self, timeout=60.0):
        """Home, ready over MQTT, wait for the vision PC's pose, show the object."""
        def run():
            q = self._joints_now()
            if np.abs(q - self.home).max() > 0.3:
                self.check_joint_move(q, self.home)
                self._move_joints(self.home, "home for the camera")
            self.runner.request_id = uuid.uuid4().hex[:8]
            self.runner.pose_timeout = float(timeout)
            while not self.runner.pose_queue.empty():
                self.runner.pose_queue.get_nowait()
            q_detect = self._joints_now()
            msg = self.runner.wait_pose(q_detect)
            T_base_object = fk(joint_map.servo_to_visual(q_detect), "camera") @ msg["T_camera_object"]
            with self.lock:
                self.object = {"T": T_base_object, "diameter": float(msg.get("diameter") or self.object_diameter),
                               "attached": False, "source": "camera"}
                self.plan = None
            self.log(f"camera: object at {np.round(T_base_object[:3, 3], 4).tolist()}")
        return self._queue("detect", run)


def _pose_from_client(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    if not np.all(np.isfinite(T)) or not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or np.linalg.det(R) < 0:
        raise TwinError("not a valid pose")
    U, _, Vt = np.linalg.svd(R)
    T[:3, :3], T[3] = U @ Vt, [0, 0, 0, 1]
    return T


# ============================================================
# WEB SERVER
# ============================================================

def make_app(twin, model, config):
    from aiohttp import WSMsgType, web

    clients = {}   # ws -> last log seq sent

    async def index(request):
        return web.FileResponse(WEB / "index.html")

    async def api_model(request):
        return web.json_response(model)

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        clients[ws] = 0
        await ws.send_json({"type": "hello", "model": model, "config": config})
        loop = asyncio.get_running_loop()
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    cmd = json.loads(msg.data)
                    if not isinstance(cmd, dict):
                        raise ValueError("a command is a JSON object")
                except ValueError as e:
                    await ws.send_json({"type": "reply", "id": None, "ok": False, "error": str(e)})
                    continue
                reply = await loop.run_in_executor(None, twin.handle, cmd)
                await ws.send_json({"type": "reply", "id": cmd.get("id"), "cmd": cmd.get("cmd"), **reply})
        finally:
            clients.pop(ws, None)
            if not clients:
                # nobody is watching any more: nothing may keep moving
                if twin.task:
                    await loop.run_in_executor(None, twin.cmd_stop)
                elif twin.armed:
                    twin.cmd_arm(False)
        return ws

    async def broadcast(app):
        while True:
            await asyncio.sleep(1 / 30)
            if not clients:
                continue
            text = json.dumps(twin.snapshot())
            for ws, seq in list(clients.items()):
                if ws.closed:
                    continue
                try:
                    await ws.send_str(text)
                    entries = twin.logs_after(seq)
                    if entries:
                        await ws.send_json({"type": "log", "entries": entries})
                        clients[ws] = entries[-1]["seq"]
                except (ConnectionError, RuntimeError):
                    pass

    async def start_broadcast(app):
        app["broadcast"] = asyncio.create_task(broadcast(app))

    async def stop_broadcast(app):
        app["broadcast"].cancel()

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/model", api_model)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/web/", WEB)
    app.on_startup.append(start_broadcast)
    app.on_cleanup.append(stop_broadcast)
    return app


def build(args):
    """Connect (or simulate) the robot, gripper and vision link; return (twin, config, cleanup)."""
    fake = None
    if args.sim:
        from fake_dobot import FakeDobot
        fake = FakeDobot(enabled=True).start()
        ip, port, feedback_port = "127.0.0.1", fake.port, fake.feedback_port
    else:
        ip, port, feedback_port = args.robot, 29999, 30004
    robot = DobotTCP(ip, port).connect()
    feedback = FeedbackReader(ip, feedback_port).start()
    feedback.wait_first(5.0)

    if args.sim or args.no_gripper:
        gripper = FakeGripper(object_position=None)
    else:
        from gripper import FeetechGripper
        side = LEFT if (args.arm or ("left" if args.robot == LEFT["robot_ip"] else "right")) == "left" else RIGHT
        gripper = FeetechGripper(args.gripper_port or side["port"], side["servo_id"] if args.gripper_id is None
                                 else args.gripper_id)

    link = None
    if args.fake_vision:
        from pick_from_pose import load_pose
        link = LocalBus()
        start_fake_vision(link, DEMO_T_CAMERA_OBJECT if args.fake_vision == "demo" else load_pose(args.fake_vision))
    elif args.mqtt:
        from mqtt_link import MqttLink
        link = MqttLink(client_id=f"digital_twin_{uuid.uuid4().hex[:6]}").connect()

    twin = Twin(robot, feedback, gripper, link, table_z=args.table_z, max_speed=args.max_speed,
                place_xy=args.place, sim=args.sim)
    if args.sim:   # something to pick right away: a tomato in front of the camera and a place spot
        twin.cmd_set_object(xy=[-0.42, 0.12])
        if twin.place is None:
            twin.place = [-0.25, 0.35]
    try:
        twin.check_model()
    except DobotError as e:
        twin.log(f"model check failed: {e}", "error")
    config = {"sim": bool(args.sim), "robot": None if args.sim else args.robot, "table_z": args.table_z,
              "detect": bool(args.mqtt or args.fake_vision), "max_speed": args.max_speed,
              "gripper": "simulated" if isinstance(gripper, FakeGripper) else "Feetech"}

    def cleanup():
        feedback.stop()
        robot.close()
        gripper.disconnect()
        if link:
            link.close()
        if fake:
            fake.stop()
    twin.fake = fake
    return twin, config, cleanup


def main():
    p = argparse.ArgumentParser(description="Nova 2 digital twin: browser app mirroring the robot both ways.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim", action="store_true", help="simulated robot (fake_dobot.py), no hardware")
    src.add_argument("--robot", metavar="IP", help="the real robot's IP, e.g. 192.168.5.1")
    p.add_argument("--table-z", type=float, help="table surface height in the base frame [m] "
                   "(required with --robot; --sim default 0 = the robot stands on the table)")
    p.add_argument("--place", type=float, nargs=2, metavar=("X", "Y"), help="default place spot [m]")
    p.add_argument("--max-speed", type=int, default=50, help="highest speed %% the app may use (default 50)")
    p.add_argument("--host", default="127.0.0.1", help="listen address (0.0.0.0 = whole network: anyone "
                   "who can open the page can move the robot)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--arm", choices=("left", "right"), help="which gripper (default as x-trainer pairs the IPs)")
    p.add_argument("--gripper-port")
    p.add_argument("--gripper-id", type=int)
    p.add_argument("--no-gripper", action="store_true", help="real robot, simulated gripper")
    p.add_argument("--mqtt", action="store_true", help="enable Detect: ready / pose with the vision PC over MQTT")
    p.add_argument("--fake-vision", metavar="POSE", help="answer Detect with this 4x4 T_camera_object file or "
                   "'demo' (with --sim only)")
    args = p.parse_args()
    if args.robot:
        if args.table_z is None:
            p.error("--table-z is required with --robot")
        if args.fake_vision:
            p.error("--fake-vision is for --sim only")
    elif args.table_z is None:
        args.table_z = 0.0
    if not 1 <= args.max_speed <= 100:
        p.error("--max-speed must be 1..100")
    if args.robot and args.arm is None and args.robot not in (LEFT["robot_ip"], RIGHT["robot_ip"]) \
            and not args.no_gripper:
        p.error(f"which gripper goes with {args.robot}? pass --arm left or --arm right")

    from aiohttp import web
    twin, config, cleanup = build(args)
    app = make_app(twin, load_model(), config)
    print(f"digital twin: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
          f"  ({'simulated robot' if args.sim else 'robot ' + args.robot})")
    try:
        web.run_app(app, host=args.host, port=args.port, print=None)
    finally:
        try:
            if twin.task:
                twin.robot.stop()
        except (DobotError, OSError):
            pass
        cleanup()


if __name__ == "__main__":
    main()
