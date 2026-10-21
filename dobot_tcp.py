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


def _checked_joints(servo_deg):
    q = np.asarray(servo_deg, dtype=float).reshape(6)
    lo, hi = SERVO_LIMITS_DEG[:, 0], SERVO_LIMITS_DEG[:, 1]
    bad = np.where((q < lo) | (q > hi) | ~np.isfinite(q))[0]
    if bad.size:
        raise DobotError("target outside the joint limits: " + ", ".join(
            f"J{i + 1}={q[i]:.2f} (limits {lo[i]:.0f}..{hi[i]:.0f})" for i in bad))
    return ",".join(f"{a:.4f}" for a in q)


def _joints_arg(servo_deg):
    return "joint={" + _checked_joints(servo_deg) + "}"


class DobotTCP:
    """protocol: "v4" (everything on port 29999, MovJ(joint={...}), Stop(),
    queue IDs), "v3" (older firmware, as x-trainer drives it: motions on
    port 30003 as JointMovJ(j1,...,j6) with SpeedJ / AccJ set on 29999,
    ResetRobot() to stop, no queue IDs) or "auto": V3 controllers open
    port 30003, V4 ones do not, so connect() can tell without moving."""

    def __init__(self, ip="192.168.5.1", port=29999, timeout=5.0, poll_s=0.05, motion_port=30003,
                 protocol="auto"):
        if protocol not in ("auto", "v3", "v4"):
            raise ValueError("protocol: auto, v3 or v4")
        self.ip, self.port, self.timeout, self.poll_s = ip, port, timeout, poll_s
        self.motion_port, self.protocol = motion_port, protocol
        self.sock = None
        self._buf = b""
        self._lock = threading.Lock()
        self.msock = None      # V3 motion port
        self._mbuf = b""
        self._mlock = threading.Lock()
        self._local_id = 0     # V3 moves have no queue ID: these stand in for it
        self._has_command_id = True
        self.stops = 0      # Stop() calls so far: a move being waited for ends at the next one
        self._pose_args = True
        self.pose_tool = 0  # what get_pose() reports: 0 = flange, None = the tool selected on the pendant
        self._stops_at_send = {}   # cmd_id -> self.stops when that move was sent

    # ------------------------------------------------------------ connection
    def connect(self):
        sock = socket.create_connection((self.ip, self.port), self.timeout)
        sock.settimeout(self.timeout)
        with self._lock:
            if self.sock is not None:   # another thread got there first
                sock.close()
            else:
                self.sock, self._buf = sock, b""
        if self.protocol in ("auto", "v3") and self.msock is None and self.motion_port:
            try:
                msock = socket.create_connection((self.ip, self.motion_port), 1.0 if self.protocol == "auto"
                                                 else self.timeout)
                msock.settimeout(self.timeout)
                with self._mlock:
                    self.msock, self._mbuf = msock, b""
                self.protocol = "v3"
            except OSError:
                if self.protocol == "v3":
                    raise
                self.protocol = "v4"
        return self

    def close(self):
        with self._lock:
            self._drop()
        with self._mlock:
            if self.msock:
                self.msock.close()
                self.msock = None

    def _drop(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    @property
    def connected(self):
        return self.sock is not None and (self.protocol != "v3" or self.msock is not None)

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def send(self, cmd):
        """Send one command, return the text inside its reply's {...}.
        Raises DobotError if ErrorID is not 0."""
        with self._lock:
            if self.sock is None:
                raise DobotError(f"{cmd}: not connected")
            try:
                self.sock.sendall(cmd.encode("ascii"))
                while b";" not in self._buf:
                    chunk = self.sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("connection closed by the robot")
                    self._buf = (self._buf + chunk).replace("；".encode(), b";")   # the guide shows both forms
            except OSError as e:
                self._drop()   # connected is False now; connect() again to go on
                raise DobotError(f"{cmd}: connection lost ({e})") from None
            reply, self._buf = self._buf.split(b";", 1)
        return self._parse(cmd, reply)

    def send_motion(self, cmd):
        """V3: a motion command on the motion port (30003)."""
        with self._mlock:
            if self.msock is None:
                raise DobotError(f"{cmd}: motion port not connected")
            try:
                self.msock.sendall(cmd.encode("ascii"))
                while b";" not in self._mbuf:
                    chunk = self.msock.recv(4096)
                    if not chunk:
                        raise ConnectionError("motion connection closed by the robot")
                    self._mbuf = (self._mbuf + chunk).replace("；".encode(), b";")
            except OSError as e:
                self.msock.close()
                self.msock = None
                raise DobotError(f"{cmd}: connection lost ({e})") from None
            reply, self._mbuf = self._mbuf.split(b";", 1)
        return self._parse(cmd, reply)

    def _parse(self, cmd, reply):
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
        """x y z [mm], rx ry rz [deg] in the base frame. Asks for the flange
        (user 0, tool 0); firmware that refuses those arguments (ErrorID
        -40001 and the like) gets plain GetPose(): the pose of the tool
        selected on the pendant, and pose_tool becomes None."""
        if self._pose_args:
            try:
                values = self._numbers("GetPose(user=0,tool=0)", 6)
                self.pose_tool = 0
                return values
            except DobotError as e:
                if e.error_id is None or e.error_id > -20000:   # not a parameter error
                    raise
                self._pose_args = False
        self.pose_tool = None
        return self._numbers("GetPose()", 6)

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
        """Stop the motion queue now (V3: ResetRobot()). A wait_motion() in
        progress ends with DobotError('... stopped') even if its move had
        not started yet."""
        self.stops += 1
        if self.protocol == "v3":
            self.send("ResetRobot()")
            return
        try:
            self.send("Stop()")
        except DobotError as e:
            if e.error_id != -10000:
                raise
            self.send("ResetRobot()")   # firmware without Stop()

    def start_drag(self):
        """Drag mode: the arm can be moved by hand (RobotMode 6)."""
        self.send("StartDrag()")

    def stop_drag(self):
        self.send("StopDrag()")

    # ------------------------------------------------------------ motion
    def mov_j(self, servo_deg, v=20, a=20):
        """Queue a joint move, return its queue ID (does not wait)."""
        stops = self.stops
        if self.protocol == "v3":
            joints = _checked_joints(servo_deg)
            self.send(f"SpeedJ({int(v)})")
            self.send(f"AccJ({int(a)})")
            self.send_motion(f"JointMovJ({joints})")
            self._local_id -= 1
            cmd_id = self._local_id
        else:
            cmd_id = int(float(self.send(f"MovJ({_joints_arg(servo_deg)},a={int(a)},v={int(v)})")))
        self._stops_at_send[cmd_id] = stops
        return cmd_id

    def mov_l(self, servo_deg, v=10, a=10):
        """Queue a straight-line move to the pose of these joints, return its
        queue ID (does not wait). V4 only: on V3 use move_l."""
        if self.protocol == "v3":
            raise DobotError("MovL to joints needs V4 firmware; move_l() follows the line with joint moves")
        stops = self.stops
        cmd_id = int(float(self.send(f"MovL({_joints_arg(servo_deg)},a={int(a)},v={int(v)})")))
        self._stops_at_send[cmd_id] = stops
        return cmd_id

    def line_waypoints(self, servo_deg, step=0.01):
        """V3 straight lines: the joints every `step` metres along the tool
        line from where the arm is now to servo_deg (orientation kept, IK
        nearest the previous point), ending exactly at servo_deg."""
        from manipulator_kinematics import IKError, fk, ik_solutions, joint_map
        start, target = self.get_angle(), np.asarray(servo_deg, dtype=float)
        T0 = fk(joint_map.servo_to_visual(start), "tool")
        T1 = fk(joint_map.servo_to_visual(target), "tool")
        n = max(1, int(np.ceil(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / step)))
        out, q = [], start
        for k in range(1, n):
            T = T0.copy()
            T[:3, 3] = T0[:3, 3] + (T1[:3, 3] - T0[:3, 3]) * k / n
            try:
                q = ik_solutions(T, q)[0]
            except IKError as e:
                raise DobotError(f"the straight line cannot be followed: {e}") from None
            out.append(q)
        return out + [target]

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
        stops = self._stops_at_send.pop(cmd_id, self.stops)
        started = False      # seen running, or past this command's ID
        while True:
            if self.stops != stops:
                raise DobotError(f"{what}: stopped")
            mode = self.robot_mode()
            if mode in (MODE_ERROR, MODE_COLLISION):
                raise DobotError(f"{what}: robot {MODE_NAMES[mode]} (alarms {self.error_ids()})", mode)
            if mode in (MODE_POWER_OFF, MODE_DISABLED):
                raise DobotError(f"{what}: robot {MODE_NAMES[mode]} during the move", mode)
            cur = self.current_command_id()
            q = self.get_angle()
            started = started or mode == MODE_RUNNING or (cur is not None and cur >= cmd_id)
            done = mode == MODE_IDLE and (cur is None or cur >= cmd_id)
            if done and reached(q):
                return q
            now = time.monotonic()
            if q_moved is None or np.abs(q - q_moved).max() > 0.05:
                t_moved, q_moved = now, q
            if not started and cur is not None and mode == MODE_IDLE and now - t_moved > 1.5:
                # idle, its ID never came up and nothing moves: the controller dropped it
                raise DobotError(f"{what}: the robot never started this move (dropped by a Stop?)")
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
        target -- because the controller picks the joints of a linear move.
        V3: joint moves along the line, 1 cm apart."""
        if self.protocol == "v3":
            q = None
            for wp in self.line_waypoints(servo_deg):
                q = self.move_j(wp, v, a, tol_deg=0.3, timeout=timeout, what=what)
            return q
        cmd_id = self.mov_l(servo_deg, v, a)
        return self.wait_motion(cmd_id, reached, timeout, what)


def main():
    parser = argparse.ArgumentParser(description="Read the robot's state over TCP (does not move it).")
    parser.add_argument("--ip", default="192.168.5.1")
    parser.add_argument("--port", type=int, default=29999)
    args = parser.parse_args()
    np.set_printoptions(precision=3, suppress=True)

    from manipulator_kinematics import fk, joint_map, lowest_point
    with DobotTCP(args.ip, args.port) as robot:
        mode = robot.robot_mode()
        q = robot.get_angle()
        p = robot.get_pose()
        model = fk(joint_map.servo_to_visual(q), "flange")[:3, 3] * 1000
        print(f"RobotMode   {mode} ({MODE_NAMES.get(mode, '?')})")
        print(f"GetAngle    {q} deg")
        print(f"GetPose     {p[:3]} mm, rx ry rz {p[3:]} deg "
              f"({'flange, tool 0' if robot.pose_tool == 0 else 'the tool selected on the pendant'})")
        print(f"model FK    {model} mm  -> differs by {np.linalg.norm(model - p[:3]):.2f} mm")
        print(f"GetErrorID  {robot.error_ids()}")
        z, part = lowest_point(joint_map.servo_to_visual(q))
        print(f"lowest      {z * 1000:.1f} mm ({part}): with the fingertips on the table, "
              f"start with --table-z {z:.3f}")


if __name__ == "__main__":
    main()
