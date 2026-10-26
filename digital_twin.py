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

Every motion and gripper step the app makes passes one gate, under the
same lock STOP takes: armed, not stopped, robot enabled and idle, feedback
fresh. STOP latches until the next deliberate Arm. The app disarms itself
when the robot is disabled, alarmed, paused, dragged, moved by someone
else, or its feedback stops; and stops a running task when no browser is
left (or none has been heard from for a few seconds).

With --sim the "robot" is fake_dobot.py (same TCP commands, same feedback
port, Nova 2 speeds), so the app is also the simulator, on the same code
path as the real arm.

Run:
    python3 digital_twin.py --sim                               # then open the URL it prints
    python3 digital_twin.py --robot 192.168.5.1 --table-z -0.05 # real arm, real gripper (left, ID 21)
    python3 digital_twin.py --robot 192.168.5.1 --table-z -0.05 --mqtt   # + "Detect" with the vision PC
"""

import argparse
import asyncio
import json
import math
import queue
import secrets
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlsplit

import numpy as np

from dobot_feedback import FeedbackReader
from dobot_tcp import MODE_IDLE, MODE_NAMES, DobotError, DobotTCP
from gripper import LEFT, RIGHT, FakeGripper, GripperError
from manipulator_kinematics import (
    SERVO_LIMITS_DEG, T_FLANGE_CAMERA, T_FLANGE_TOOL, IKError, TableConstraint, fk, ik_solutions,
    joint_map, lowest_point, pose, rot_z,
)
from mqtt_link import COMMAND_TOPIC, STATUS_TOPIC, LocalBus, MessageError, camera_pose, matrix_quaternion, parse_command
from manipulator_agent import ManipulatorAgent, plan_place
from pick_runner import Move, PickFailed, PickRunner, measured_flange, start_fake_vision

WEB = Path(__file__).resolve().parent / "twin_web"
URDF = WEB / "model" / "nova2" / "nova2_robot.urdf"
T_LINK6_FLANGE = pose(rot_z(math.pi / 2), [0.0, 0.0, 0.0])   # Dobot's link6 -> the DH flange frame
HOME = joint_map.home_servo_deg
LINE_STEP = 0.005        # [m] straight-line moves are checked every 5 mm
LINE_MAX_JUMP = 10.0     # [deg] largest joint change per step on a line (more = IK branch switch)
LINE_ROT_TOL = 0.1       # [deg] a straight-line move keeps the orientation (within reading noise)
FEEDBACK_STALE = 0.5     # [s] older feedback = robot not connected
START_TOL = 0.2          # [deg] a checked move only starts from where it was checked
DISARM_AFTER = 0.15      # [s] a reason to disarm must last this long (one stale packet is not one)
CLIENT_SILENT = 3.0      # [s] while a task runs, some browser must have been heard from this recently


class TwinError(RuntimeError):
    pass


class Stopped(TwinError):
    """STOP, a disarm or a fault ended the task before this step."""


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
# INPUT CHECKS (a browser can send anything)
# ============================================================

def _num(value, name, lo=-math.inf, hi=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TwinError(f"{name} must be a finite number")
    if not lo <= value <= hi:
        raise TwinError(f"{name} must be {lo:g}..{hi:g}")
    return float(value)


def _vec(value, n, name):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise TwinError(f"{name} must be {n} numbers")
    return [_num(v, f"{name}[{i}]") for i, v in enumerate(value)]


def _bool(value, name):
    if not isinstance(value, bool):
        raise TwinError(f"{name} must be true or false")
    return value


def _pose_from_client(T):
    if not isinstance(T, (list, tuple)) or len(T) != 4:
        raise TwinError("a pose is a 4x4 matrix")
    T = np.array([_vec(row, 4, "pose row") for row in T], dtype=float)
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or np.linalg.det(R) < 0:
        raise TwinError("not a valid pose")
    U, _, Vt = np.linalg.svd(R)
    T[:3, :3], T[3] = U @ Vt, [0, 0, 0, 1]
    return T


def _rotation_deg(R_a, R_b):
    R = R_a.T @ R_b
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


# ============================================================
# THE ROBOT AND GRIPPER AS THE TWIN'S TASKS SEE THEM
# ============================================================

class GuardedRobot:
    """The robot for the twin's tasks and its pick runner: every motion
    first passes Twin.gate() under Twin.motion_lock -- the lock STOP takes
    to send Stop() -- so a move is either on its way before the Stop() or
    never sent. Everything else (GetAngle, GetPose, ...) goes straight through."""

    def __init__(self, robot, twin):
        self._robot, self._twin = robot, twin

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def _send(self, send, servo_deg, v, a):
        with self._twin.motion_lock:
            self._twin.gate()
            self._twin.in_motion = True
            cmd_id = send(servo_deg, v, a)
            self._twin.moved = True
        return cmd_id

    def _wait(self, cmd_id, reached, timeout, what):
        try:
            return self._robot.wait_motion(cmd_id, reached, timeout, what)
        finally:
            self._twin.in_motion = False
            self._twin.motion_ended = time.monotonic()

    def mov_j(self, servo_deg, v=20, a=20):
        return self._send(self._robot.mov_j, servo_deg, v, a)

    def mov_l(self, servo_deg, v=10, a=10):
        return self._send(self._robot.mov_l, servo_deg, v, a)

    def move_j(self, servo_deg, v=20, a=20, tol_deg=0.3, timeout=60.0, what="MovJ"):
        target = np.asarray(servo_deg, dtype=float)
        cmd_id = self.mov_j(target, v, a)
        return self._wait(cmd_id, lambda q: np.abs(q - target).max() <= tol_deg, timeout, what)

    def move_l(self, servo_deg, reached, v=10, a=10, timeout=60.0, what="MovL"):
        if self._robot.protocol == "v3":   # no MovL to joints: joint moves 1 cm apart, each through the gate
            q = None
            for wp in self._robot.line_waypoints(servo_deg):
                q = self.move_j(wp, v, a, tol_deg=0.3, timeout=timeout, what=what)
            return q
        cmd_id = self.mov_l(servo_deg, v, a)
        return self._wait(cmd_id, reached, timeout, what)


class TwinGripper:
    """Wraps the real (or fake) gripper: each action passes the twin's gate,
    and the twin learns the result (and carries the object). A fake gripper
    (--sim, --no-gripper) only closes on the object if the tool tip is
    really at it."""

    def __init__(self, inner, twin):
        self.inner, self.twin = inner, twin

    def _gate(self):
        with self.twin.motion_lock:
            self.twin.gate()

    def open(self, speed=100, wait=True):
        self._gate()
        self.inner.open(speed, wait)
        self.twin._gripper_changed(0, False)

    def close(self, speed=50):
        self._gate()
        if isinstance(self.inner, FakeGripper):
            held = self.twin.grip["holding"]
            self.inner.object_position = (self.twin.grip["position"] if held else
                                          140 if self.twin._object_between_fingers() else None)
        result = self.inner.close(speed)
        self.twin._gripper_changed(result.position, result.holding)
        return result

    def position(self):
        """Where the fingers are now (0 open .. 255 closed), if the gripper can say."""
        if hasattr(self.inner, "read_raw"):
            return self.inner.from_raw(self.inner.read_raw())
        return getattr(self.inner, "position", 0)

    def disconnect(self):
        self.inner.disconnect()


# ============================================================
# THE TWIN
# ============================================================

def now_minus(obj, attr):
    """Seconds since obj.<attr> was last stamped by this call (inf the first time); stamps it."""
    now = time.monotonic()
    last = getattr(obj, attr, -math.inf)
    if now - last > 1.0:
        setattr(obj, attr, now)
    return now - last


class Twin:
    def __init__(self, robot, feedback, gripper, link=None, table_z=0.0, home=None, max_speed=50,
                 place_xy=None, sim=False, object_diameter=0.06, log=print):
        self.robot, self.feedback = robot, feedback
        self.sim, self.table_z = sim, table_z
        self.home = np.asarray(HOME if home is None else home, dtype=float)
        self.max_speed = int(max_speed)
        self.speed = min(20, self.max_speed)
        self.lock = threading.RLock()        # the twin's state
        self.motion_lock = threading.Lock()  # gate + send vs Stop()
        self.armed = False
        self.task = self.step = None
        self.cancel = threading.Event()      # STOP latch, cleared only by Arm
        self.moved = False                   # the running task has sent a motion
        self.in_motion = False               # a MovJ/MovL of ours is on its way or running
        self.motion_ended = -math.inf
        self.stop_gen = 0                    # +1 at every STOP / disarm
        self._ctx = threading.local()        # the stop_gen a command was received under
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
        self.can_detect = link is not None
        self.link = link or LocalBus()
        self.guarded = GuardedRobot(robot, self)
        self.gripper = TwinGripper(gripper, self)
        try:
            self.grip["position"] = int(self.gripper.position())
            self.grip["holding"] = 10 < self.grip["position"] < 245   # unknown: treat as holding
        except (GripperError, OSError) as e:
            self.log(f"gripper position unknown: {e}", "warn")
        self.runner = PickRunner(self.guarded, self.gripper, self.link, home=self.home, table_z=table_z,
                                 log=lambda text: self.log(text, "info"))
        self.runner.cancel = self.cancel
        self.runner.on_status = lambda state, detail: self._set_step(f"{state}" + (f": {detail}" if detail else ""))
        self.fake = None            # the simulated controller, with --sim
        self.clients_heard = None   # set by the web server: seconds since any browser spoke
        self._tasks = queue.Queue()
        threading.Thread(target=self._worker, daemon=True, name="twin-tasks").start()
        threading.Thread(target=self._watch, daemon=True, name="twin-watch").start()
        if link is not None:   # the vision agent's commands (and yours, by hand: mqtt_send.py)
            self._commands = self.link.subscribe(COMMAND_TOPIC)
            threading.Thread(target=self._command_loop, daemon=True, name="twin-mqtt-commands").start()

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
            "lowest": [round(v, 4) if isinstance(v, float) else v for v in lowest_point(joint_map.servo_to_visual(q))],
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
                "stopped": self.cancel.is_set(), "task": self.task, "step": self.step,
                "speed": self.speed, "max_speed": self.max_speed,
                "gripper": dict(self.grip), "object": obj, "place": self.place, "table_z": self.table_z,
                "ghost": None if self.ghost is None else np.round(self.ghost, 4).tolist(),
                "plan": self.plan, "model_error_mm": self.model_error_mm, "sim": self.sim,
                "can_detect": self.can_detect, "log_seq": self._log_seq,
            }

    # ------------------------------------------------------------ MQTT commands
    def _command_loop(self):
        while True:
            payload = self._commands.get()
            try:
                self.mqtt_command(payload)
            except Exception as e:   # a bad message must not end the loop
                self.log(f"MQTT command: unexpected {type(e).__name__}: {e}", "error")

    def _publish_status(self, state, cmd, detail=""):
        msg = {"state": state, "action": cmd.get("action"), "product": cmd.get("product", ""),
               "detail": str(detail), "stamp": time.time()}
        if cmd.get("request_id") is not None:
            msg["request_id"] = cmd["request_id"]
        try:
            self.link.publish(STATUS_TOPIC, msg)
        except Exception as e:
            self.log(f"MQTT status not sent: {e}", "warn")

    def mqtt_command(self, payload):
        """One command from the vision agent (MQTT_COMMAND_TOPIC):
        stop -> STOP; idle -> the object moves to position (nothing moves);
        start pick -> object at position, pick it and hold it at home;
        start place -> put the held object down at position.
        Moving needs the twin armed in a browser, as from the page."""
        gen = self.stop_gen   # as received: a STOP after this cancels it
        try:
            cmd = parse_command(payload)
        except MessageError as e:
            self.log(f"MQTT command refused: {e}", "warn")
            self._publish_status("refused", {}, e)
            return
        what = " ".join(v for v in (cmd["state"], cmd["action"], cmd["product"]) if v)
        where = "" if cmd["position"] is None else f" at {np.round(cmd['position'][:3], 4).tolist()} (camera)"
        if cmd["state"] != "idle" or now_minus(self, "_idle_logged") > 1.0:   # a stream of idle poses: once a second
            self.log(f"MQTT command: {what}{where}")
        if cmd["state"] == "stop":
            self.request_stop()
            self.cmd_stop()
            self._publish_status("stopped", cmd)
            return
        position = cmd["position"]

        def run(c):
            r = self.handle(c, gen=gen, on_done=lambda state, detail: self._publish_status(state, cmd, detail))
            if not r["ok"]:
                if cmd["state"] != "idle" or now_minus(self, "_idle_refused_logged") > 1.0:
                    self.log(f"MQTT {what}: {r['error']}", "warn")
                self._publish_status("refused", cmd, r["error"])
            return r

        if cmd["state"] == "idle" or (cmd["action"] == "pick" and position is not None):
            if position is not None:
                r = run({"cmd": "set_object_camera", "position": position, "orientation": cmd["orientation"],
                         "symmetric": cmd["symmetric"], "product": cmd["product"]})
                if not r["ok"]:
                    return
                if cmd["state"] == "idle":
                    self._publish_status("updated", cmd, f"object at {r['xyz']} (base frame), "
                                                         f"quaternion {r['quaternion']} (base frame)")
            elif cmd["state"] == "idle":
                self._publish_status("idle", cmd)
            if cmd["state"] == "idle":
                return
        if cmd["action"] == "pick":
            r = run({"cmd": "pick", "hold": True})
        else:
            r = run({"cmd": "place_at", "position": position})
        if r["ok"]:
            self._publish_status("accepted", cmd, r.get("queued", ""))

    # ------------------------------------------------------------ the gate
    def gate(self):
        """Raise Stopped unless the app may move the arm now. Called under
        motion_lock right before every motion or gripper step."""
        if self.cancel.is_set() or not self.armed:
            raise Stopped("stopped")
        if self.feedback.age() > FEEDBACK_STALE:
            raise Stopped("no feedback from the robot")
        mode = self.robot.robot_mode()
        if mode != MODE_IDLE:
            raise Stopped(f"the robot is {MODE_NAMES.get(mode, mode)}")

    def request_stop(self):
        """The STOP latch, at once and without robot I/O."""
        self.cancel.set()
        with self.lock:
            self.armed = False
            self.stop_gen += 1

    def _sent_before_stop(self):
        gen = getattr(self._ctx, "gen", None)
        return gen is not None and gen != self.stop_gen

    def _send_stop(self):
        """Stop() after any move already through the gate; reconnect once
        if the command link is down. Returns whether it reached the robot."""
        with self.motion_lock:
            for attempt in (1, 2):
                try:
                    if not self.robot.connected:
                        self.robot.connect()
                    self.robot.stop()
                    return True
                except (DobotError, OSError) as e:
                    if attempt == 2:
                        self.log(f"Stop() did not reach the robot ({e}): use the emergency stop", "error")
        return False

    def _abort(self, why):
        """A fault: latch, disarm, and stop the robot if a task was moving it."""
        busy = self.task is not None
        self.request_stop()
        if busy:
            self._send_stop()
        self.log(f"{why} -- disarmed", "error")

    def _watch(self):
        """Disarm when the robot is no longer the app's to move: disabled,
        alarm, collision, paused, drag mode, moving without the app, or
        silent; stop a task no browser is watching any more."""
        since = {}
        q_ref = None
        while True:
            time.sleep(0.05)
            try:
                q_ref = self._watch_once(since, q_ref)
            except Exception as e:   # the watcher must not die: fail safe
                since.clear()
                q_ref = None
                if self.armed:
                    self._abort(f"watcher error {type(e).__name__}: {e}")

    def _watch_once(self, since, q_ref):
            if not self.armed:
                since.clear()
                return None
            why = None
            s, age = self.feedback.latest(), self.feedback.age()
            heard = self.clients_heard() if self.clients_heard else 0.0
            if s is None or age > FEEDBACK_STALE:
                why = "no feedback from the robot"
            elif s["drag_status"] or s["robot_mode"] == 6:
                why = "the robot is in drag mode"
            elif s["error_status"] or s["collision_state"] or s["robot_mode"] in (9, 11):
                why = f"the robot is {MODE_NAMES.get(int(s['robot_mode']), 'in alarm')}"
            elif s["robot_mode"] in (1, 2, 3, 4):
                why = f"the robot is {MODE_NAMES.get(int(s['robot_mode']))}"
            elif s["robot_mode"] == 10:
                why = "the robot was paused"
            elif s["robot_mode"] == 8:
                why = "someone is jogging the robot"
            elif s["robot_mode"] == 7 and not self.in_motion and time.monotonic() - self.motion_ended > 0.3:   # feedback lags our own end
                why = "the robot is moving, but not for this app"
            elif self.task is not None and heard > CLIENT_SILENT:
                why = "no browser heard from"
            immediate = why is not None and why != "no feedback from the robot" and why != "no browser heard from"
            if s is not None and age <= FEEDBACK_STALE:
                q = np.asarray(s["q_actual"], dtype=float)
                if self.in_motion or q_ref is None or time.monotonic() - self.motion_ended < 0.3:
                    q_ref = q
                elif why is None and np.abs(q - q_ref).max() > 0.1:
                    why, immediate = "the robot was moved, but not by this app", True
            now = time.monotonic()
            for key in list(since):
                if key != why:
                    del since[key]
            if why is None:
                return q_ref
            since.setdefault(why, now)
            if immediate or now - since[why] >= DISARM_AFTER:
                self._abort(why if why != "no browser heard from" else f"no browser heard from for {heard:.1f} s")
                since.clear()
                return None
            return q_ref

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
        one branch, above the table. The line keeps the tool's orientation.
        Returns (joints at the end, the end pose actually used)."""
        T_from = fk(joint_map.servo_to_visual(q_from), "tool")
        if _rotation_deg(T_from[:3, :3], T_to[:3, :3]) > LINE_ROT_TOL:
            raise TwinError("a straight-line move here keeps the tool's orientation; use a joint move to turn it")
        T_to = T_to.copy()
        T_to[:3, :3] = T_from[:3, :3]
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
        return q, T_to

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
        tip = fk(joint_map.servo_to_visual(self.robot.get_angle()), "tool")[:3, 3]
        return float(np.linalg.norm(tip - center)) <= max(0.012, 0.35 * d)

    def _gripper_changed(self, position, holding):
        with self.lock:   # the state first: nothing below may leave it stale
            self.grip = {"position": int(position), "holding": bool(holding)}
        T_tool = fk(joint_map.servo_to_visual(self.robot.get_angle()), "tool")
        with self.lock:
            if holding:
                if self.object is None or \
                        (not self.object["attached"] and
                         np.linalg.norm(self.object["T"][:3, 3] - T_tool[:3, 3]) > max(0.03, self.object["diameter"])):
                    # something the twin did not know about: show it in the fingers
                    self.object = {"T": T_tool.copy(), "diameter": self.object_diameter, "source": "gripped"}
                if not self.object.get("attached"):
                    self.object["attached"] = True
                    self.object["T_tool_obj"] = np.linalg.inv(T_tool) @ self.object["T"]
            elif self.object is not None and self.object.get("attached"):
                self.object["attached"] = False
                T = T_tool @ self.object["T_tool_obj"]   # where the fingers let it go
                if self.table_z is not None:            # ... and it ends up on the table
                    T[2, 3] = self.table_z + self.object["diameter"] / 2
                self.object["T"] = T

    # ------------------------------------------------------------ tasks
    def _queue(self, name, fn, q_from=None):
        """Run fn on the task thread. q_from: the joints the move was checked
        from; it only starts if the arm is still there."""
        with self.lock:
            if self.task is not None:
                raise TwinError(f"busy: {self.task}")
            if not self.armed or self.cancel.is_set():
                raise TwinError("not armed: switch on 'Arm' to let the app move the robot")
            if self._sent_before_stop():
                raise TwinError("STOP was pressed after this command was sent: not run")
            self.task, self.step = name, None
        self._tasks.put((name, fn, q_from, getattr(self._ctx, "on_done", None)))
        return {"queued": name}

    def _worker(self):
        while True:
            name, fn, q_from, on_done = self._tasks.get()
            self.moved = False
            self.log(f"{name}: started")
            result = ["failed", ""]
            try:
                if self.cancel.is_set() or not self.armed:
                    raise Stopped("stopped before it started")
                if q_from is not None and np.abs(self.robot.get_angle() - q_from).max() > START_TOL:
                    raise TwinError("the arm moved since this move was checked: try again")
                fn()
                self.log(f"{name}: done")
                result = ["done", ""]
            except PickFailed as e:              # missed grip: back home, or nothing moved
                self.log(f"{name}: {e}", "warn")
                result[1] = str(e)
            except (IKError, TwinError) as e:
                result[1] = str(e)
                if self.moved or isinstance(e, Stopped):
                    if not self.cancel.is_set():
                        self._abort(f"{name}: {e}")
                    else:
                        self.log(f"{name}: {e}", "warn")
                else:
                    self.log(f"{name}: {e}", "warn")   # refused before any motion
            except (DobotError, GripperError, OSError) as e:
                result[1] = str(e)
                if self.cancel.is_set() and "stopped" in str(e):
                    self.log(f"{name}: stopped", "warn")
                else:
                    self._abort(f"{name}: {e}")
            except Exception as e:   # a bug must not kill the worker
                result[1] = f"unexpected {type(e).__name__}: {e}"
                self._abort(f"{name}: unexpected {type(e).__name__}: {e}")
            finally:
                with self.lock:
                    self.task = None
                    self.ghost = None
                    self.plan = None
                if self.cancel.is_set() and result[0] != "done":
                    result[0] = "stopped"
                if on_done is not None:
                    try:
                        on_done(*result)
                    except Exception as e:
                        self.log(f"{name}: reporting the result failed: {e}", "warn")

    def _speeds(self):
        self.runner.speed = self.speed
        self.runner.line_speed = max(1, self.speed // 2)

    def _move_joints(self, q, what):
        self._speeds()
        self.guarded.move_j(q, v=self.runner.speed, a=self.runner.speed, what=what)

    def _move_line(self, q, T, what):
        self._speeds()
        self.guarded.move_l(q, self.runner._reached(T), v=self.runner.line_speed, a=self.runner.line_speed, what=what)

    # ------------------------------------------------------------ pick & place
    def _object_message(self, q_now):
        with self.lock:
            if self.object is None:
                raise TwinError("no object: click 'Set object' and then the table, or press Detect")
            if self.object["attached"] or self.grip["holding"]:
                raise TwinError("the gripper is holding something: open it (or place it) first")
            T_base_object = self.object["T"].copy()
            d = self.object["diameter"]
            symmetric = self.object.get("symmetric", True)
        T_camera_object = np.linalg.inv(fk(joint_map.servo_to_visual(q_now), "camera")) @ T_base_object
        return {"T_camera_object": T_camera_object, "origin": "center", "diameter": d,
                "symmetric": symmetric, "object": "twin", "request_id": None}

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
    def handle(self, cmd, gen=None, on_done=None):
        """One command from a browser (or MQTT) -> reply dict. Motions are
        queued; the reply says whether they were accepted, the state shows
        the rest. on_done(state, detail) is called when a queued task ends."""
        self._ctx.gen = gen
        self._ctx.on_done = on_done
        try:
            if not isinstance(cmd, dict):
                raise TwinError("a command is a JSON object")
            name = cmd.get("cmd")
            args = {k: v for k, v in cmd.items() if k not in ("cmd", "id")}
            fn = getattr(self, f"cmd_{name}", None) if isinstance(name, str) else None
            if fn is None:
                raise TwinError(f"unknown command {name!r}")
            if name not in ("stop", "arm") and not self.robot.connected:
                # the command port dropped: disarm, then reconnect (a lost motion is never re-sent)
                if self.armed:
                    self._abort("command connection to the robot lost")
                self.robot.connect()
                self.log("command connection to the robot re-opened", "warn")
            return {"ok": True, **(fn(**args) or {})}
        except (TwinError, IKError, MessageError, DobotError, GripperError, OSError) as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:   # bad arguments must not cost the browser its connection
            return {"ok": False, "error": f"bad command: {type(e).__name__}: {e}"}

    def cmd_arm(self, on=True):
        if not _bool(on, "on"):
            return self.cmd_stop()   # taking the permission away stops what it allowed (no check-then-act)
        if not self.robot_state()["connected"]:
            raise TwinError("no feedback from the robot")
        if not self.robot.connected:
            self.robot.connect()
        mode = self.robot.robot_mode()   # the controller's own word, not feedback up to 8 ms old
        if mode != MODE_IDLE:
            raise TwinError(f"the robot is {MODE_NAMES.get(mode, mode)}: enable it first")
        if self.table_z is None:
            raise TwinError("no table height (--table-z)")
        if self.model_error_mm is None:
            self.check_model()
        if self.model_error_mm > 3.0:
            raise TwinError(f"the kinematic model is {self.model_error_mm:.1f} mm off the controller: not arming")
        bad = self._table().violations(joint_map.servo_to_visual(self.robot.get_angle()))
        if bad:   # the arm is where the table should be: the table height is wrong
            raise TwinError(f"the arm is already below the table height {self.table_z:.3f} m ({', '.join(bad)}): "
                            "check --table-z (put the fingertips on the table and read 'Lowest point')")
        with self.lock:
            if self.task is not None:
                raise TwinError(f"busy: {self.task}")
            if self._sent_before_stop():
                raise TwinError("STOP was pressed after this Arm was sent: Arm again")
            self.cancel.clear()
            self.armed = True
        self.log("armed: the app may move the robot", "warn")
        return {}

    def cmd_stop(self):
        self.request_stop()
        sent = self._send_stop()
        self.log("STOP: disarmed" + (", Stop() sent" if sent else ""), "error")
        return {"sent": sent}

    def cmd_keepalive(self):
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
        self.request_stop()
        self.robot.disable()
        self.log("robot disabled")
        return {}

    def cmd_clear_error(self):
        self.robot.clear_error()
        self.log("alarms cleared")
        return {}

    def cmd_drag(self, on=True):
        on = _bool(on, "on")
        with self.lock:
            if self.task:
                raise TwinError(f"busy: {self.task}")
        self.request_stop()   # nothing from the app moves the arm while a hand does
        if on:
            self.robot.start_drag()
            self.log("drag mode: move the arm by hand, the twin follows", "warn")
        else:
            self.robot.stop_drag()
            self.log("drag mode off")
        return {}

    def cmd_speed(self, value):
        value = _num(value, "speed", 1, self.max_speed)
        with self.lock:
            self.speed = int(value)
        return {}

    def check_model(self):
        q = self.robot.get_angle()
        measured, frame = measured_flange(self.robot, q)
        model = fk(joint_map.servo_to_visual(q), "flange")[:3, 3]
        self.model_error_mm = float(np.linalg.norm(measured - model) * 1000)
        self.log(f"model check: controller ({frame}) vs FK {self.model_error_mm:.2f} mm"
                 + ("" if self.model_error_mm <= 3.0 else
                    " -- on the pendant select tool 0 (flange) or the 197 mm tool, then Enable again"),
                 "info" if self.model_error_mm <= 3.0 else "error")
        return self.model_error_mm

    # previews (no motion, allowed any time)
    def cmd_preview_joints(self, joints):
        q_to = self.check_joint_move(self._joints_now(), _vec(joints, 6, "joints"))
        with self.lock:
            self.ghost = q_to
        return {"joints": q_to.tolist()}

    def cmd_preview_pose(self, T, linear=False):
        T = _pose_from_client(T)
        q_now = self._joints_now()
        q = self.check_line(q_now, T)[0] if _bool(linear, "linear") else self.check_joint_move(q_now, self.ik(T, q_now))
        with self.lock:
            self.ghost = q
        return {"joints": np.round(q, 4).tolist()}

    def cmd_clear_preview(self):
        with self.lock:
            self.ghost = None
        return {}

    def cmd_clear_plan(self):
        with self.lock:
            self.ghost = None
            self.plan = None
        return {}

    # motions (armed, one at a time)
    def cmd_move_joints(self, joints):
        q_from = self._joints_now()
        q_to = self.check_joint_move(q_from, _vec(joints, 6, "joints"))
        with self.lock:
            self.ghost = q_to
        return self._queue("move joints", lambda: self._move_joints(q_to, "move joints"), q_from)

    def cmd_jog_joint(self, joint, delta):
        i = joint if isinstance(joint, int) and not isinstance(joint, bool) else -1
        if not 0 <= i < 6:
            raise TwinError("jog: joint must be 0..5")
        q = self._joints_now()
        q[i] += _num(delta, "jog", -30, 30)
        return self.cmd_move_joints(q.tolist())

    def cmd_move_pose(self, T, linear=False):
        T = _pose_from_client(T)
        q_from = self._joints_now()
        if _bool(linear, "linear"):
            q, T = self.check_line(q_from, T)
            with self.lock:
                self.ghost = q
            return self._queue("move line", lambda: self._move_line(q, T, "move line"), q_from)
        q = self.check_joint_move(q_from, self.ik(T, q_from))
        with self.lock:
            self.ghost = q
        return self._queue("move to pose", lambda: self._move_joints(q, "move to pose"), q_from)

    def cmd_jog_tool(self, axis, delta):
        """Straight-line jog of the tool tip along a base axis, delta in mm."""
        if axis not in ("x", "y", "z"):
            raise TwinError("jog: axis x / y / z")
        delta = _num(delta, "jog", -100, 100)
        T = fk(joint_map.servo_to_visual(self._joints_now()), "tool")
        T[{"x": 0, "y": 1, "z": 2}[axis], 3] += delta / 1000
        return self.cmd_move_pose(T.tolist(), linear=True)

    def cmd_home(self):
        return self.cmd_move_joints(self.home.tolist())

    def cmd_gripper(self, action):
        if action not in ("open", "close"):
            raise TwinError("gripper: open or close")
        return self._queue(f"gripper {action}", lambda: self.gripper.open() if action == "open" else self.gripper.close())

    # the scene
    def cmd_set_object(self, xyz=None, xy=None, yaw_deg=0.0, diameter=None):
        d = self.object_diameter if diameter is None else _num(diameter, "diameter", 0.01, 0.2)
        if xy is not None:
            if self.table_z is None:
                raise TwinError("no table height")
            x, y = _vec(xy, 2, "xy")
            xyz = [x, y, self.table_z + d / 2]
        elif xyz is None:
            raise TwinError("give xy (on the table) or xyz")
        xyz = _vec(xyz, 3, "xyz")
        if max(abs(v) for v in xyz) > 2.0:
            raise TwinError("the object is more than 2 m from the robot")
        yaw = _num(yaw_deg, "yaw_deg", -360, 360)
        with self.lock:
            if (self.object is not None and self.object["attached"]) or self.grip["holding"]:
                raise TwinError("the gripper is holding the object")
            self.object = {"T": pose(rot_z(math.radians(yaw)), xyz), "diameter": d,
                           "attached": False, "source": "manual"}
            self.plan = None
        return {}

    def cmd_set_object_camera(self, position, orientation=None, product="", diameter=None, symmetric=None):
        """The object's pose as the vision agent (FoundationPose) measures it,
        in the wrist camera's frame seen from where the arm is now: position
        [x, y, z] m (or [x, y, z, qx, qy, qz, qw], or [x, y, z, w] with w a
        turn about the view axis in degrees) and orientation [qx, qy, qz, qw].
        The twin's object takes that position and orientation. symmetric:
        the grasp ignores the rotation (default: when none was given)."""
        T_cam_obj, rotated = camera_pose(position, orientation)
        if not isinstance(product, str):
            raise TwinError("product must be a string")
        symmetric = (not rotated) if symmetric is None else _bool(symmetric, "symmetric")
        d = self.object_diameter if diameter is None else _num(diameter, "diameter", 0.01, 0.2)
        with self.lock:
            if self.task is not None:
                raise TwinError(f"busy: {self.task} (the camera's view moves with the arm)")
            if (self.object is not None and self.object["attached"]) or self.grip["holding"]:
                raise TwinError("the gripper is holding the object")
        state = self.robot_state()
        if state.get("mode") == 7:
            raise TwinError("the arm is moving: the camera's view moves with it")
        q = self._joints_now()
        T = fk(joint_map.servo_to_visual(q), "camera") @ T_cam_obj
        if max(abs(v) for v in T[:3, 3]) > 2.0:
            raise TwinError("the object is more than 2 m from the robot")
        with self.lock:
            self.object = {"T": T, "diameter": d, "attached": False, "symmetric": symmetric,
                           "source": f"camera{': ' + product[:64] if product else ''}"}
            self.plan = None
        xyz = T[:3, 3]
        tilt = math.degrees(math.acos(float(np.clip(T[2, 2], -1.0, 1.0))))   # object z axis from straight up
        yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        reply = {"xyz": np.round(xyz, 4).tolist(), "quaternion": np.round(matrix_quaternion(T[:3, :3]), 5).tolist(),
                 "tilt_deg": round(tilt, 1), "yaw_deg": round(yaw, 1)}
        detail = f", turned {yaw:.0f} deg, its z axis {tilt:.0f} deg from straight up" if rotated else ""
        if self.table_z is not None:
            above = float(xyz[2] - self.table_z)
            reply["above_table_mm"] = round(above * 1000, 1)
            detail += f", centre {above * 1000:.0f} mm above the table (a {d * 1000:.0f} mm object on it: {d * 500:.0f})"
        now = time.monotonic()
        last_t, last_xyz = getattr(self, "_camera_logged", (-math.inf, None))
        if now - last_t > 1.0 or last_xyz is None or np.linalg.norm(xyz - last_xyz) > 0.005:   # a live stream: not every frame
            self._camera_logged = (now, xyz.copy())
            self.log(f"object from the camera {np.round(T_cam_obj[:3, 3], 4).tolist()} m: "
                     f"base {np.round(xyz, 4).tolist()} m{detail}")
        return reply

    def cmd_clear_object(self):
        with self.lock:
            if self.object is not None and self.object["attached"]:
                raise TwinError("the gripper is holding the object")
            self.object = self.plan = None
        return {}

    def cmd_set_place(self, xy=None):
        place = None if xy is None else _vec(xy, 2, "xy")
        if place is not None and max(abs(v) for v in place) > 2.0:
            raise TwinError("the place spot is more than 2 m from the robot")
        with self.lock:
            self.place = place
            self.plan = None
        return {}

    def cmd_plan(self):
        return {"plan": self.plan_pick()}

    def cmd_pick(self, hold=False):
        """Pick the object, then place it at the place spot (if set) or,
        with hold=True, keep holding it at home (MQTT action pick)."""
        place = None if _bool(hold, "hold") else self.place
        q_now = self._joints_now()
        _, moves, _, _ = self.runner.plan(self._object_message(q_now), q_now, place)   # refuse now if it can't be done
        with self.lock:
            self.plan = self._plan_view(moves, q_now)

        def run():
            self.gripper.open()   # as pick_runner does: never come down on it with closed fingers
            q_now = self.robot.get_angle()
            msg = self._object_message(q_now)
            _, moves, _, _ = self.runner.plan(msg, q_now, place)
            with self.lock:
                self.plan = self._plan_view(moves, q_now)
            self._speeds()
            self.runner.execute(moves, place)
        return self._queue("pick and place" if place else "pick", run)

    def _place_moves(self, q_now, xy):
        """Put the held object down with its centre at xy, on the table:
        pre-place (MovJ), down (MovL), open, back up (MovL), home (MovJ)."""
        with self.lock:
            if not self.grip["holding"]:
                raise TwinError("not holding anything to place")
            obj = self.object if self.object is not None and self.object.get("attached") else None
            T_tool_obj = obj["T_tool_obj"].copy() if obj else np.eye(4)
            d = obj["diameter"] if obj else self.object_diameter
        if self.table_z is None:
            raise TwinError("no table height")
        T_tool = fk(joint_map.servo_to_visual(q_now), "tool")
        centre = (T_tool @ T_tool_obj)[:3, 3]
        T_put = T_tool.copy()
        T_put[:3, 3] += np.array([xy[0], xy[1], self.table_z + d / 2]) - centre
        agent = ManipulatorAgent(table_z=self.table_z, **self.runner._agent_kwargs())
        table = agent.table_for(np.eye(4))
        try:
            line = plan_place(agent, table, T_put, q_now)
        except IKError as e:
            raise TwinError(f"place spot {np.round(xy, 4).tolist()}: {e}") from None
        (_, T_pre, q_pre), (_, T_down, q_down) = line[0], line[-1]
        bad = table.path_violations(joint_map.servo_to_visual(q_pre), joint_map.servo_to_visual(self.home))
        if bad:
            raise TwinError("the way back home goes below the table: " + ", ".join(bad))
        return [Move("pre_place", "movj", q_pre, T_pre),
                Move("place", "movl", q_down, T_down, then="open"),
                Move("retreat", "movl", q_pre, T_pre),
                Move("home", "movj", self.home.copy(), fk(joint_map.servo_to_visual(self.home), "tool"))]

    def cmd_place_at(self, position=None, xy=None):
        """Put the held object down at xy (base frame, m) or at position
        [x, y, z, w] in the camera's frame seen from where the arm is now
        (only its x, y are used: it goes on the table)."""
        if (position is None) == (xy is None):
            raise TwinError("give position (camera frame) or xy (base frame)")
        q_now = self._joints_now()
        if position is not None:
            xy = (fk(joint_map.servo_to_visual(q_now), "camera") @ camera_pose(position)[0])[:2, 3].tolist()
        xy = [float(v) for v in _vec(xy, 2, "xy")]
        if max(abs(v) for v in xy) > 2.0:
            raise TwinError("the place spot is more than 2 m from the robot")
        moves = self._place_moves(q_now, xy)
        with self.lock:
            self.plan = self._plan_view(moves, q_now)

        def run():
            q = self.robot.get_angle()
            moves = self._place_moves(q, xy)
            with self.lock:
                self.plan = self._plan_view(moves, q)
            self._speeds()
            self.runner.execute(moves, xy)
        return self._queue("place", run, q_now)

    def cmd_detect(self, timeout=60.0):
        """Home, ready over MQTT, wait for the vision PC's pose, show the object."""
        if not self.can_detect:
            raise TwinError("no vision link: start with --mqtt (or --fake-vision in --sim)")
        timeout = _num(timeout, "timeout", 1, 300)
        if self.grip["holding"]:
            raise TwinError("the gripper is holding something: open it (or place it) first")

        def run():
            q = self.robot.get_angle()
            if np.abs(q - self.home).max() > 0.3:
                self.check_joint_move(q, self.home)
                self._move_joints(self.home, "home for the camera")
            self.runner.request_id = uuid.uuid4().hex[:8]
            self.runner.pose_timeout = timeout
            while not self.runner.pose_queue.empty():
                self.runner.pose_queue.get_nowait()
            q_detect = self.robot.get_angle()
            msg = self.runner.wait_pose(q_detect)
            if np.abs(self.robot.get_angle() - q_detect).max() > 0.05:
                raise TwinError("the arm moved while the camera was looking: press Detect again")
            T_base_object = fk(joint_map.servo_to_visual(q_detect), "camera") @ msg["T_camera_object"]
            with self.lock:
                self.object = {"T": T_base_object, "diameter": float(msg.get("diameter") or self.object_diameter),
                               "attached": False, "source": "camera"}
                self.plan = None
            self.log(f"camera: object at {np.round(T_base_object[:3, 3], 4).tolist()}")
        return self._queue("detect", run)


# ============================================================
# WEB SERVER
# ============================================================

def _no_constants(name):
    raise ValueError(f"{name} is not a number")


def make_app(twin, model, config, token):
    """The twin's web app. Browsers must open it with ?token=... (printed at
    start) and from its own address: a page from anywhere else can't reach
    the WebSocket that moves the robot."""
    from aiohttp import WSMsgType, web

    clients = {}                                        # ws -> {"seq": last log sent, "heard": monotonic}
    last_heard = [-math.inf]                            # any browser, monotonic (one float: no dict walk off-loop)
    instance = secrets.token_hex(4)                     # a restarted server starts its log afresh
    work = ThreadPoolExecutor(4, thread_name_prefix="twin-cmd")     # ordinary commands
    stops = ThreadPoolExecutor(1, thread_name_prefix="twin-stop")   # STOP never waits behind them

    def heard():
        if not clients:
            return math.inf
        return time.monotonic() - last_heard[0]
    twin.clients_heard = heard

    async def index(request):
        return web.FileResponse(WEB / "index.html")

    async def api_model(request):
        return web.json_response(model)

    def allowed(request):
        origin = request.headers.get("Origin")
        if origin is not None and urlsplit(origin).netloc != request.host:
            return False
        return secrets.compare_digest(request.query.get("token", "").encode(), token.encode())

    async def stop_now(loop):
        twin.request_stop()                        # latched at once, on the event loop
        return await loop.run_in_executor(stops, twin.cmd_stop)

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=2.0, max_msg_size=64 * 1024)
        await ws.prepare(request)
        if not allowed(request):
            await ws.send_json({"type": "denied", "error": "open the address digital_twin.py printed (with its token)"})
            await ws.close(code=4003, message=b"denied")
            return ws
        loop = asyncio.get_running_loop()
        clients[ws] = {"seq": 0, "heard": time.monotonic()}
        last_heard[0] = time.monotonic()
        inbox = asyncio.Queue()
        await ws.send_json({"type": "hello", "model": model, "config": config, "log_seq": twin._log_seq,
                            "instance": instance})

        async def reply(cmd, result):
            cid = cmd.get("id")
            if not (isinstance(cid, str) and len(cid) <= 64 or isinstance(cid, int) and not isinstance(cid, bool)
                    and abs(cid) < 2 ** 53):
                cid = None
            name = cmd.get("cmd") if isinstance(cmd.get("cmd"), str) else None
            if not ws.closed:
                await ws.send_json({"type": "reply", "id": cid, "cmd": name, **result},
                                   dumps=lambda o: json.dumps(o, allow_nan=False, default=str))

        async def run_commands():   # this browser's commands, one at a time
            while True:
                cmd, gen = await inbox.get()
                result = await loop.run_in_executor(work, twin.handle, cmd, gen)
                try:
                    await reply(cmd, result)
                except (ValueError, TypeError) as e:   # never lose this browser's later commands
                    await reply(cmd, {"ok": False, "error": f"reply not sent: {e}"})

        runner = asyncio.create_task(run_commands())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                clients[ws]["heard"] = last_heard[0] = time.monotonic()
                try:
                    cmd = json.loads(msg.data, parse_constant=_no_constants)
                    if not isinstance(cmd, dict):
                        raise ValueError("a command is a JSON object")
                except (ValueError, RecursionError) as e:
                    await reply({}, {"ok": False, "error": f"bad message: {e}"})
                    continue
                if cmd.get("cmd") == "keepalive":
                    continue
                if cmd.get("cmd") == "stop":
                    await reply(cmd, {"ok": True, **await stop_now(loop)})
                elif cmd.get("cmd") == "arm" and cmd.get("on") is False:
                    await reply(cmd, {"ok": True, **await stop_now(loop)})
                else:
                    inbox.put_nowait((cmd, twin.stop_gen))
        finally:
            runner.cancel()
            clients.pop(ws, None)
            if not clients:
                # nobody is watching any more: nothing may keep moving
                if twin.task is not None:
                    await stop_now(loop)
                    twin.log("the last browser closed: stopped", "error")
                elif twin.armed:
                    twin.request_stop()
                    twin.log("the last browser closed: disarmed", "warn")
        return ws

    async def broadcast(app):
        while True:
            await asyncio.sleep(1 / 30)
            if not clients:
                continue
            try:
                text = json.dumps(twin.snapshot(), allow_nan=False)
            except ValueError as e:   # never send what a browser can't parse
                twin.log(f"state not sent: {e}", "error")
                await asyncio.sleep(1)
                continue
            for ws, c in list(clients.items()):
                if ws.closed:
                    continue
                try:
                    await ws.send_str(text)
                    entries = twin.logs_after(c["seq"])
                    if entries:
                        await ws.send_json({"type": "log", "entries": entries})
                        c["seq"] = entries[-1]["seq"]
                except (ConnectionError, RuntimeError):
                    pass

    async def start_broadcast(app):
        app["broadcast"] = asyncio.create_task(broadcast(app))

    async def stop_broadcast(app):
        app["broadcast"].cancel()
        work.shutdown(wait=False)
        stops.shutdown(wait=False)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/model", api_model)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/web/", WEB)
    app.on_startup.append(start_broadcast)
    app.on_cleanup.append(stop_broadcast)
    return app


def demo_pose_on_table(table_z, xy=(-0.42, 0.12), diameter=0.06):
    """--fake-vision demo: a tomato seen from home, resting on the twin's table."""
    T_base_object = pose(np.eye(3), [xy[0], xy[1], table_z + diameter / 2])
    return np.linalg.inv(fk(np.zeros(6), "camera")) @ T_base_object


def build(args):
    """Connect (or simulate) the robot, gripper and vision link; return (twin, config, cleanup)."""
    fake = None
    if args.sim:
        from fake_dobot import FakeDobot
        fake = FakeDobot(enabled=True, protocol="v3" if args.protocol == "v3" else "v4").start()
        ip, port, feedback_port = "127.0.0.1", fake.port, fake.feedback_port
        motion_port, protocol = fake.motion_port or 30003, fake.protocol
    else:
        ip, port, feedback_port = args.robot, 29999, 30004
        motion_port, protocol = 30003, args.protocol
    robot = DobotTCP(ip, port, motion_port=motion_port, protocol=protocol).connect()
    print(f"robot {ip}: TCP protocol {robot.protocol}")
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
        start_fake_vision(link, demo_pose_on_table(args.table_z) if args.fake_vision == "demo"
                          else load_pose(args.fake_vision))
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
              "detect": twin.can_detect, "max_speed": args.max_speed,
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
    p.add_argument("--protocol", choices=("auto", "v3", "v4"), default="auto",
                   help="controller TCP protocol: v3 = motion on port 30003 (JointMovJ, ResetRobot), "
                        "v4 = all on 29999 (MovJ(joint=...), Stop); auto (default) picks v3 if 30003 is open")
    p.add_argument("--pedestal", action="store_true",
                   help="the robot stands on a pedestal, so the table is below its base (--table-z below -0.02)")
    p.add_argument("--max-speed", type=int, default=50, help="highest speed %% the app may use (default 50)")
    p.add_argument("--host", default="127.0.0.1", help="listen address (0.0.0.0 = whole network: anyone "
                   "with the printed address, token included, can move the robot)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--token", help="access token in the address (default: a new random one each start)")
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
        if args.table_z < -0.02 and not args.pedestal:
            p.error(f"--table-z {args.table_z} puts the table {-args.table_z * 100:.0f} cm below the robot's base "
                    "(z = 0 is the base's mounting face). If the robot stands on the table, the table is at about "
                    "0: measure it (README, 'Table height'). If it really stands on a pedestal, add --pedestal")
    elif args.table_z is None:
        args.table_z = 0.0
    if not 1 <= args.max_speed <= 100:
        p.error("--max-speed must be 1..100")
    if args.robot and args.arm is None and args.robot not in (LEFT["robot_ip"], RIGHT["robot_ip"]) \
            and not args.no_gripper:
        p.error(f"which gripper goes with {args.robot}? pass --arm left or --arm right")
    if args.token is not None and len(args.token) < 8:
        p.error("--token: at least 8 characters")

    from aiohttp import web
    token = args.token or secrets.token_urlsafe(12)
    twin, config, cleanup = build(args)
    app = make_app(twin, load_model(), config, token)
    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "::") else f"[{args.host}]" if ":" in args.host else args.host
    print(f"digital twin ({'simulated robot' if args.sim else 'robot ' + args.robot}): "
          f"open http://{host}:{args.port}/?token={quote(token, safe='')}")
    try:
        web.run_app(app, host=args.host, port=args.port, print=None)
    finally:
        try:
            if twin.task:
                twin.request_stop()
                twin.robot.stop()
        except (DobotError, OSError):
            pass
        cleanup()


if __name__ == "__main__":
    main()
