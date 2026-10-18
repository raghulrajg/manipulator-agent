"""
dobot_tcp.py
-------------
Dobot Nova 2 over TCP/IP, controller V4 ("TCP_IP Remote Control Interface
Guide V4.6.0" in Docs/). Everything goes through the dashboard port 29999:
each command is sent as text and answered with "ErrorID,{values},Cmd();".

The controller plans every motion itself, so the PC only sends targets:
    MovJ(joint={j1,...,j6})   joint move: all joints start and stop together
    MovL(joint={j1,...,j6})   straight line of the tool; the controller works
                              out the joints on the way
and then watches the robot until the move is done:
    GetCurrentCommandID()     queue ID of the motion running now
    RobotMode()               5 enabled and idle, 7 running, 9 alarm, 11 collision
    GetAngle()                joint angles [deg]
    GetPose(user=0,tool=0)    flange x y z [mm] and rx ry rz [deg], from the controller

Joint angles here are always the controller's own ("servo") degrees, the
same numbers the pendant shows and joint_map.servo_to_visual takes.

Run (read only, the arm does not move):
    python3 dobot_tcp.py --ip 192.168.5.1
"""

import argparse
import re
import socket
import threading
import time

import numpy as np

from manipulator_kinematics import SERVO_LIMITS_DEG

# RobotMode() values (guide, RobotMode)
MODE_POWER_OFF = 3
MODE_DISABLED = 4
MODE_IDLE = 5          # enabled and idle
MODE_RUNNING = 7
MODE_ERROR = 9         # uncleared alarm (has priority over every other mode)
MODE_PAUSED = 10
MODE_COLLISION = 11
MODE_NAMES = {
    1: "initializing", 2: "brake released", 3: "powered off", 4: "disabled",
    5: "enabled and idle", 6: "drag mode", 7: "running", 8: "jogging",
    9: "in alarm", 10: "paused", 11: "stopped by collision detection",
}

# ErrorID values (guide, chapter 4 "Error Code")
ERROR_TEXT = {
    -1: "command received but not executed",
    -2: "robot in alarm status: clear the alarm and send it again",
    -3: "robot in emergency stop: release the e-stop, clear the alarm",
    -4: "robot powered off",
    -5: "a script is running: pause or stop it first",
    -7: "a script is paused: stop it first",
    -8: "robot certification expired",
    -10000: "command does not exist",
    -20000: "wrong number of parameters",
}

_REPLY = re.compile(r"\s*(-?\d+)\s*,\s*\{(.*?)\}\s*,", re.S)


def error_text(err):
    if err in ERROR_TEXT:
        return ERROR_TEXT[err]
    for base, what in ((-30000, "required parameter type"), (-40000, "required parameter range"),
                       (-50000, "optional parameter type"), (-60000, "optional parameter range")):
        if base - 10 < err < base:
            return f"wrong {what} (parameter {base - err})"
    return "see the guide's Error Code chapter"


class DobotError(RuntimeError):
    def __init__(self, message, error_id=None):
        super().__init__(message)
        self.error_id = error_id


def _joints_arg(servo_deg):
    q = np.asarray(servo_deg, dtype=float).reshape(6)
    lo, hi = SERVO_LIMITS_DEG[:, 0], SERVO_LIMITS_DEG[:, 1]
    bad = np.where((q < lo) | (q > hi) | ~np.isfinite(q))[0]
    if bad.size:
        raise DobotError("target outside the joint limits: " + ", ".join(
            f"J{i + 1}={q[i]:.2f} (limits {lo[i]:.0f}..{hi[i]:.0f})" for i in bad))
    return "joint={" + ",".join(f"{a:.4f}" for a in q) + "}"


class DobotTCP:
    def __init__(self, ip="192.168.5.1", port=29999, timeout=5.0, poll_s=0.05):
        self.ip, self.port, self.timeout, self.poll_s = ip, port, timeout, poll_s
        self.sock = None
        self._buf = b""
        self._lock = threading.Lock()
        self._has_command_id = True

    # ------------------------------------------------------------ connection
    def connect(self):
        self.sock = socket.create_connection((self.ip, self.port), self.timeout)
        self.sock.settimeout(self.timeout)
        self._buf = b""
        return self

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def send(self, cmd):
        """Send one command, return the text inside its reply's {...}.
        Raises DobotError if ErrorID is not 0."""
        if self.sock is None:
            raise DobotError("not connected")
        with self._lock:
            self.sock.sendall(cmd.encode("ascii"))
            while b";" not in self._buf:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise DobotError("connection closed by the robot")
                self._buf = (self._buf + chunk).replace("；".encode(), b";")   # the guide shows both forms
            reply, self._buf = self._buf.split(b";", 1)
        text = reply.decode("utf-8", errors="ignore")
        m = _REPLY.match(text)
        if not m:
            raise DobotError(f"unparseable reply to {cmd}: {text!r}")
        err = int(m.group(1))
        if err != 0:
            raise DobotError(f"{cmd} -> ErrorID {err}: {error_text(err)}", err)
        return m.group(2)

    def _numbers(self, cmd, n):
        values = [float(v) for v in self.send(cmd).split(",") if v.strip()]
        if len(values) < n:
            raise DobotError(f"{cmd} returned {len(values)} values, expected {n}")
        return np.array(values[:n])

    # ------------------------------------------------------------- state
    def request_control(self):
        """Switch the controller to TCP mode (only allowed while disabled)."""
        self.send("RequestControl()")

    def clear_error(self):
        self.send("ClearError()")

    def enable(self, timeout=20.0):
        if self.robot_mode() == MODE_IDLE:
            return
        self.clear_error()
        self.send("EnableRobot()")
        t0 = time.monotonic()
        while (mode := self.robot_mode()) != MODE_IDLE:
            if time.monotonic() - t0 > timeout:
                raise DobotError(f"robot did not enable, it is {MODE_NAMES.get(mode, mode)}")
            time.sleep(0.1)

    def disable(self):
        self.send("DisableRobot()")

    def robot_mode(self):
        return int(float(self.send("RobotMode()")))

    def error_ids(self):
        """GetErrorID: '[[controller alarms],[J1],...,[J6]]', [] = none."""
        return self.send("GetErrorID()").strip()

    def get_angle(self):
        """Joint angles [deg] as the controller reports them."""
        return self._numbers("GetAngle()", 6)

    def get_pose(self):
        """Flange (tool 0) in the base frame (user 0): x y z [mm], rx ry rz [deg]."""
        return self._numbers("GetPose(user=0,tool=0)", 6)

    def current_command_id(self):
        """Queue ID of the motion command running now; None if this
        firmware has no GetCurrentCommandID."""
        if not self._has_command_id:
            return None
        try:
            return int(float(self.send("GetCurrentCommandID()")))
        except DobotError as e:
            if e.error_id != -10000:
                raise
            self._has_command_id = False
            return None

    def speed_factor(self, percent):
        self.send(f"SpeedFactor({int(percent)})")

    def stop(self):
        """Stop the motion queue now."""
        self.send("Stop()")

    # ------------------------------------------------------------ motion
    def mov_j(self, servo_deg, v=20, a=20):
        """Queue a joint move, return its queue ID (does not wait)."""
        return int(float(self.send(f"MovJ({_joints_arg(servo_deg)},a={int(a)},v={int(v)})")))

    def mov_l(self, servo_deg, v=10, a=10):
        """Queue a straight-line move to the pose of these joints, return its
        queue ID (does not wait)."""
        return int(float(self.send(f"MovL({_joints_arg(servo_deg)},a={int(a)},v={int(v)})")))

    def wait_motion(self, cmd_id, reached, timeout=60.0, what="move", max_wait=1800.0):
        """
        Poll until motion cmd_id is finished (its ID is current and the robot
        is idle) and reached(joints) is True; return the joints. Raises
        DobotError on an alarm, collision, disable, if the robot stops
        somewhere else, or if no joint has moved 0.05 deg for timeout seconds
        (a slow move is fine as long as it keeps moving; a pause counts as
        stopped; reading noise does not count as moving), or after max_wait
        seconds in any case.
        """
        t0 = t_moved = time.monotonic()
        q_moved = None
        stopped_since = None
        while True:
            mode = self.robot_mode()
            if mode in (MODE_ERROR, MODE_COLLISION):
                raise DobotError(f"{what}: robot {MODE_NAMES[mode]} (alarms {self.error_ids()})", mode)
            if mode in (MODE_POWER_OFF, MODE_DISABLED):
                raise DobotError(f"{what}: robot {MODE_NAMES[mode]} during the move", mode)
            cur = self.current_command_id()
            q = self.get_angle()
            done = mode == MODE_IDLE and (cur is None or cur >= cmd_id)
            if done and reached(q):
                return q
            now = time.monotonic()
            if q_moved is None or np.abs(q - q_moved).max() > 0.05:
                t_moved, q_moved = now, q
            if done and cur is not None:
                # finished but not there: give the last reading a moment, then fail
                stopped_since = stopped_since or now
                if now - stopped_since > 1.0:
                    raise DobotError(f"{what}: the robot stopped at {np.round(q, 3).tolist()}, "
                                     "not at the target")
            else:
                stopped_since = None
            if now - t_moved > timeout or now - t0 > max_wait:
                raise DobotError(f"{what}: no joint has moved for {now - t_moved:.0f} s, {now - t0:.0f} s in all, "
                                 f"and it is not at the target "
                                 f"(robot {MODE_NAMES.get(mode, mode)}, joints {np.round(q, 3).tolist()})")
            time.sleep(self.poll_s)

    def move_j(self, servo_deg, v=20, a=20, tol_deg=0.3, timeout=60.0, what="MovJ"):
        """MovJ and wait until every joint is within tol_deg of the target."""
        target = np.asarray(servo_deg, dtype=float)
        cmd_id = self.mov_j(target, v, a)
        return self.wait_motion(cmd_id, lambda q: np.abs(q - target).max() <= tol_deg, timeout, what)

    def move_l(self, servo_deg, reached, v=10, a=10, timeout=60.0, what="MovL"):
        """MovL and wait until reached(joints) -- e.g. the tool tip is at the
        target -- because the controller picks the joints of a linear move."""
        cmd_id = self.mov_l(servo_deg, v, a)
        return self.wait_motion(cmd_id, reached, timeout, what)


def main():
    parser = argparse.ArgumentParser(description="Read the robot's state over TCP (does not move it).")
    parser.add_argument("--ip", default="192.168.5.1")
    parser.add_argument("--port", type=int, default=29999)
    args = parser.parse_args()
    np.set_printoptions(precision=3, suppress=True)

    from manipulator_kinematics import fk, joint_map
    with DobotTCP(args.ip, args.port) as robot:
        mode = robot.robot_mode()
        q = robot.get_angle()
        p = robot.get_pose()
        model = fk(joint_map.servo_to_visual(q), "flange")[:3, 3] * 1000
        print(f"RobotMode   {mode} ({MODE_NAMES.get(mode, '?')})")
        print(f"GetAngle    {q} deg")
        print(f"GetPose     {p[:3]} mm, rx ry rz {p[3:]} deg (flange, tool 0)")
        print(f"model FK    {model} mm  -> differs by {np.linalg.norm(model - p[:3]):.2f} mm")
        print(f"GetErrorID  {robot.error_ids()}")


if __name__ == "__main__":
    main()
