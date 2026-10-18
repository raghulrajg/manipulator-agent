"""
fake_dobot.py
--------------
A stand-in for the Nova 2 controller on localhost, for pick_runner.py
--dry-run and the tests: it answers the V4 TCP commands dobot_tcp.py sends
(same reply format, queue IDs, RobotMode values) and moves its joints in a
background thread. GetPose is the flange FK of manipulator_kinematics.

MovJ and MovL both move linearly in joint space here (the real MovL is a
straight tool line; the pick only uses it on lines the planner checked).

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

from manipulator_kinematics import fk, joint_map

_CMD = re.compile(r"\s*(\w+)\((.*?)\)", re.S)


def euler_xyz_deg(R):
    """Rotation -> Dobot rx ry rz [deg] (fixed axes, X then Y then Z)."""
    ry = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    rx = math.atan2(R[2, 1], R[2, 2])
    rz = math.atan2(R[1, 0], R[0, 0])
    return np.degrees([rx, ry, rz])


class FakeDobot:
    def __init__(self, joints=None, deg_per_s=120.0, command_id=True, fault=None):
        self.joints = np.array(joint_map.home_servo_deg if joints is None else joints, dtype=float)
        self.deg_per_s = deg_per_s
        self.command_id = command_id
        self.fault = fault
        self.mode = 4            # disabled until EnableRobot
        self.queue = []          # [(id, name, target, v)]
        self.current_id = 0
        self.next_id = 0
        self.motions = 0
        self.log = []            # [(command name, args text, joints when received)]
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(4)
        self.port = self._server.getsockname()[1]

    def start(self):
        threading.Thread(target=self._accept, daemon=True).start()
        threading.Thread(target=self._motion, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        self._server.close()

    # ------------------------------------------------------------ network
    def _accept(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

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
                    conn.sendall(self.handle(m.group(1), m.group(2)).encode("ascii"))

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
            T = fk(joint_map.servo_to_visual(self.joints), "flange")
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
            _, _, target, v = job
            kind, n = self.fault or (None, None)
            fault_here = n == self.motions
            if fault_here and kind == "short":
                target = target - np.sign(target - start) * 1.0
            duration = max(0.02, np.abs(target - start).max() / (self.deg_per_s * v / 100))
            t0 = time.monotonic()
            while True:
                s = min(1.0, (time.monotonic() - t0) / duration)
                with self.lock:
                    if self.mode != 7:          # stopped from outside
                        break
                    stuck = fault_here and kind == "stuck" and s >= 0.5
                    if stuck:
                        flip = 1 if int(time.monotonic() * 200) % 2 else -1
                        self.joints = start + 0.5 * (target - start) + 0.02 * flip
                    else:
                        self.joints = start + s * (target - start)
                    if fault_here and kind == "collision" and s >= 0.5:
                        self.mode, self.queue = 11, []
                        break
                    if s >= 1.0 and not stuck:
                        self.mode = 5
                        break
                time.sleep(0.005)
