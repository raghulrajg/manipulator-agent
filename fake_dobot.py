"""
fake_dobot.py
--------------
A stand-in for the Nova 2 controller on localhost, for pick_runner.py
--dry-run, the digital twin's simulation mode and the tests:

  - dashboard port: answers the V4 TCP commands dobot_tcp.py sends (same
    reply format, queue IDs, RobotMode values)
  - feedback port: a 1440-byte real-time packet every 8 ms, like port 30004
    (dobot_feedback.py), with the actual joints, mode, tool pose, queue ID
  - motion in a background thread at Nova 2 speed: 135 deg/s per joint at
    100 % (Nova User Guide, Appendix A) x the command's v x SpeedFactor,
    with smooth start and stop. MovJ is joint-interpolated, MovL moves the
    tool along a straight line (IK at every step), as on the robot.
  - StartDrag / StopDrag, and external_move() for a robot moved by hand or
    from the pendant: the feedback shows it, the commands don't know it.

GetPose and ToolVectorActual are the flange (tool 0) from manipulator_kinematics.

Faults for the tests: fault=("collision", n) stops the n-th motion halfway
in RobotMode 11; fault=("short", n) ends it 1 deg short of its target;
fault=("stuck", n) leaves it halfway, still "running", with its joint
readings jittering 0.02 deg (until Stop).
"""

import math
import re
import socket
import threading
import time

import numpy as np

from dobot_feedback import pack_feedback
from manipulator_kinematics import T_FLANGE_TOOL, IKError, fk, ik_solutions, joint_map

_CMD = re.compile(r"\s*(\w+)\((.*?)\)", re.S)
NOVA2_DEG_PER_S = 135.0
NOVA2_MM_PER_S = 1600.0


def euler_xyz_deg(R):
    """Rotation -> Dobot rx ry rz [deg] (fixed axes, X then Y then Z)."""
    ry = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    rx = math.atan2(R[2, 1], R[2, 2])
    rz = math.atan2(R[1, 0], R[0, 0])
    return np.degrees([rx, ry, rz])


def _smooth(s):
    """0..1 -> 0..1 with zero speed at both ends (like the controller's ramps)."""
    return s * s * (3.0 - 2.0 * s)


class FakeDobot:
    def __init__(self, joints=None, deg_per_s=NOVA2_DEG_PER_S, command_id=True, fault=None,
                 feedback=True, enabled=False, pose_args=True, pendant_tool=None):
        """pose_args=False: GetPose(user=..,tool=..) is refused with -40001, as
        on some V4 firmware. pendant_tool: a 4x4 flange->tool transform for
        the tool selected on the pendant (None = tool 0, the flange)."""
        self.pose_args = pose_args
        self.pendant_tool = np.eye(4) if pendant_tool is None else np.asarray(pendant_tool, dtype=float)
        self.joints = np.array(joint_map.home_servo_deg if joints is None else joints, dtype=float)
        self.target = self.joints.copy()
        self.deg_per_s = deg_per_s
        self.command_id = command_id
        self.fault = fault
        self.mode = 5 if enabled else 4   # disabled until EnableRobot
        self.speed_factor = 100
        self.queue = []          # [(id, name, target, v)]
        self.current_id = 0
        self.next_id = 0
        self.motions = 0
        self.log = []            # [(command name, args text, joints when received)]
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._server = self._listen()
        self.port = self._server.getsockname()[1]
        self._feedback = self._listen() if feedback else None
        self.feedback_port = self._feedback.getsockname()[1] if feedback else None

    @staticmethod
    def _listen():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen(8)
        return s

    def start(self):
        threading.Thread(target=self._accept, args=(self._server, self._serve), daemon=True).start()
        if self._feedback:
            threading.Thread(target=self._accept, args=(self._feedback, self._stream), daemon=True).start()
        threading.Thread(target=self._motion, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        self._server.close()
        if self._feedback:
            self._feedback.close()

    # ------------------------------------------------------------ outside world
    def external_move(self, joints, duration=0.5, wait=True):
        """The robot moved without a command from us (pendant jog, hand
        guiding in drag mode). Only while it is idle or in drag mode."""
        with self.lock:
            if self.mode not in (5, 6):
                raise RuntimeError(f"cannot move by hand in mode {self.mode}")
            start, target, was = self.joints.copy(), np.asarray(joints, dtype=float), self.mode
            if was == 5:
                self.mode = 8          # single move (jog)

        def run():
            t0 = time.monotonic()
            while True:
                s = min(1.0, (time.monotonic() - t0) / max(duration, 1e-3))
                with self.lock:
                    self.joints = start + _smooth(s) * (target - start)
                    self.target = self.joints.copy()
                    if s >= 1.0:
                        if self.mode == 8:
                            self.mode = 5
                        return
                time.sleep(0.005)

        if wait:
            run()
        else:
            threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------ network
    def _accept(self, server, handler):
        while not self._stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                return
            threading.Thread(target=handler, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        buf = ""
        with conn:
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                except OSError:
                    return
                if not data:
                    return
                buf += data.decode("ascii")
                while m := _CMD.match(buf):
                    buf = buf[m.end():]
                    try:
                        conn.sendall(self.handle(m.group(1), m.group(2)).encode("ascii"))
                    except OSError:
                        return

    def _stream(self, conn):
        """Port 30004: one packet every 8 ms."""
        with conn:
            next_t = time.monotonic()
            while not self._stop.is_set():
                try:
                    conn.sendall(self.feedback_packet())
                except OSError:
                    return
                next_t += 0.008
                time.sleep(max(0.0, next_t - time.monotonic()))

    def feedback_packet(self):
        with self.lock:
            q, target, mode, cid, sf = self.joints.copy(), self.target.copy(), self.mode, self.current_id, self.speed_factor
        T = fk(joint_map.servo_to_visual(q), "flange")
        return pack_feedback(
            robot_mode=mode, timestamp_ms=int(time.time() * 1000), speed_scaling=sf / 100.0,
            q_actual=q, q_target=target,
            tool_vector_actual=[*(T[:3, 3] * 1000), *euler_xyz_deg(T[:3, :3])],
            enable_status=int(mode in (5, 6, 7, 8, 10)), drag_status=int(mode == 6),
            running_status=int(mode == 7), error_status=int(mode in (9, 11)),
            collision_state=int(mode == 11), current_command_id=cid)

    # ------------------------------------------------------------ commands
    def handle(self, name, args):
        with self.lock:
            self.log.append((name, args, self.joints.copy()))
            err, values = self._run(name, args)
        return f"{err},{{{values}}},{name}({args});"

    def _run(self, name, args):
        if name == "RequestControl":
            return (-1, "") if self.mode == 7 else (0, "")
        if name == "ClearError":
            if self.mode in (9, 11):
                self.mode = 5
            return 0, ""
        if name == "EnableRobot":
            self.mode = 5 if self.mode in (4, 5) else self.mode
            return 0, ""
        if name == "DisableRobot":
            self.mode, self.queue = 4, []
            return 0, ""
        if name == "RobotMode":
            return 0, str(self.mode)
        if name == "GetAngle":
            return 0, ",".join(f"{a:.4f}" for a in self.joints)
        if name == "GetPose":
            if args.strip() and not self.pose_args:
                return -40001, ""
            T = fk(joint_map.servo_to_visual(self.joints), "flange")
            if not args.strip():
                T = T @ self.pendant_tool
            return 0, ",".join(f"{v:.4f}" for v in [*(T[:3, 3] * 1000), *euler_xyz_deg(T[:3, :3])])
        if name == "GetErrorID":
            alarm = "[22]" if self.mode == 11 else "[]"
            return 0, f"[{alarm},[],[],[],[],[],[]]"
        if name == "GetCurrentCommandID":
            return (0, str(self.current_id)) if self.command_id else (-10000, "")
        if name == "Stop":
            self.queue = []
            self.mode = 5 if self.mode == 7 else self.mode
            return 0, ""
        if name == "SpeedFactor":
            m = re.fullmatch(r"\s*(\d+)\s*", args)
            if not m or not 1 <= int(m.group(1)) <= 100:
                return -40001, ""
            self.speed_factor = int(m.group(1))
            return 0, ""
        if name == "StartDrag":
            if self.mode != 5:
                return -1, ""
            self.mode = 6
            return 0, ""
        if name == "StopDrag":
            if self.mode == 6:
                self.mode = 5
            return 0, ""
        if name in ("MovJ", "MovL"):
            if self.mode in (9, 11):
                return -2, ""
            if self.mode not in (5, 7):
                return -1, ""
            m = re.search(r"joint=\{([^}]*)\}", args)
            v = re.search(r"\bv=(\d+)", args)
            if not m:
                return -30001, ""
            target = np.array([float(x) for x in m.group(1).split(",")])
            if target.shape != (6,):
                return -40001, ""
            self.next_id += 1
            self.queue.append((self.next_id, name, target, int(v.group(1)) if v else 100))
            return 0, str(self.next_id)
        return -10000, ""

    # ------------------------------------------------------------ motion
    def _line(self, start, target):
        """MovL: joints along the straight tool line from start to target,
        as a function of 0..1 (IK nearest the previous step)."""
        T0 = fk(joint_map.servo_to_visual(start), "flange")
        T1 = fk(joint_map.servo_to_visual(target), "flange")
        R0, R1 = T0[:3, :3], T1[:3, :3]
        same_rotation = np.allclose(R0, R1, atol=1e-6)
        prev = [start.copy()]

        def at(s):
            if s >= 1.0 or not same_rotation:   # the pick only uses constant-orientation lines
                q = start + s * (target - start)
            else:
                T = T0.copy()
                T[:3, 3] = T0[:3, 3] + s * (T1[:3, 3] - T0[:3, 3])
                try:
                    q = ik_solutions(T @ T_FLANGE_TOOL, prev[0])[0]
                except IKError:
                    q = start + s * (target - start)
            prev[0] = q
            return q
        return at, float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])) * 1000

    def _motion(self):
        while not self._stop.is_set():
            with self.lock:
                job = self.queue.pop(0) if self.queue and self.mode == 5 else None
                if job:
                    self.mode, self.current_id = 7, job[0]
                    self.motions += 1
                    start = self.joints.copy()
            if not job:
                time.sleep(0.005)
                continue
            _, name, target, v = job
            kind, n = self.fault or (None, None)
            fault_here = n == self.motions
            if fault_here and kind == "short":
                target = target - np.sign(target - start) * 1.0
            with self.lock:
                self.target = target.copy()
            ratio = max(v, 1) / 100 * self.speed_factor / 100
            duration = np.abs(target - start).max() / (self.deg_per_s * ratio)
            path = lambda s: start + _smooth(s) * (target - start)   # noqa: E731
            if name == "MovL":
                line, mm = self._line(start, target)
                duration = max(duration, mm / (NOVA2_MM_PER_S * ratio))
                path = lambda s: line(_smooth(s))   # noqa: E731
            duration = max(0.02, duration)
            t0 = time.monotonic()
            while True:
                s = min(1.0, (time.monotonic() - t0) / duration)
                q = path(s)
                with self.lock:
                    if self.mode != 7:          # stopped from outside
                        break
                    stuck = fault_here and kind == "stuck" and s >= 0.5
                    if stuck:
                        flip = 1 if int(time.monotonic() * 200) % 2 else -1
                        self.joints = path(0.5) + 0.02 * flip
                    else:
                        self.joints = target.copy() if s >= 1.0 else q
                    if fault_here and kind == "collision" and s >= 0.5:
                        self.mode, self.queue = 11, []
                        break
                    if s >= 1.0 and not stuck:
                        self.mode = 5
                        break
                time.sleep(0.005)
