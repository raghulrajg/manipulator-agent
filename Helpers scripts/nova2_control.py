import re
import socket
import time

import numpy as np

from joint_mapping import joint_map, build_dh_params  # noqa: F401  (re-exported)

# Nova 2 servo-angle limits [deg] (Nova User Guide, Appendix A)
NOVA2_LIMITS_DEG = np.array([
    [-360, 360],   # J1
    [-180, 180],   # J2
    [-156, 156],   # J3
    [-360, 360],   # J4
    [-360, 360],   # J5
    [-360, 360],   # J6
], dtype=float)

# RobotMode() values used here (TCP/IP guide, RobotMode)
MODE_ENABLE = 5      # enabled and idle
MODE_ERROR = 9
MODE_COLLISION = 11


class NovaError(RuntimeError):
    pass


class NovaClient:
    def __init__(self, ip="192.168.5.1", port=29999, jmap=joint_map,
                 limits_deg=NOVA2_LIMITS_DEG, timeout=5.0):
        self.ip, self.port, self.timeout = ip, port, timeout
        self.map = jmap
        self.limits = np.asarray(limits_deg, dtype=float)
        self.sock = None

    # ------------------------------------------------------------ connection
    def connect(self):
        self.sock = socket.create_connection((self.ip, self.port), self.timeout)
        self.sock.settimeout(self.timeout)
        return self

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def _send(self, cmd):
        """Send one command, return the list of values inside {...} as strings."""
        self.sock.sendall(cmd.encode("ascii"))
        buf = b""
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise NovaError("connection closed by robot")
            buf += chunk
            text = buf.decode("utf-8", errors="ignore")
            if text.rstrip().endswith((";", "；")):   # the guide shows both forms
                break
        m = re.match(r"\s*(-?\d+)\s*,\s*\{(.*?)\}\s*,", text, re.S)
        if not m:
            raise NovaError(f"unparseable reply to {cmd!r}: {text!r}")
        err = int(m.group(1))
        if err != 0:
            raise NovaError(f"{cmd} failed, ErrorID={err}")
        return [v.strip() for v in m.group(2).split(",") if v.strip()]

    # ------------------------------------------------------------- state
    def request_control(self):
        self._send("RequestControl()")

    def clear_error(self):
        self._send("ClearError()")

    def enable(self, wait=15.0):
        self.clear_error()
        self._send("EnableRobot()")
        self.wait_mode(MODE_ENABLE, wait)

    def disable(self):
        self._send("DisableRobot()")

    def robot_mode(self):
        return int(self._send("RobotMode()")[0])

    def wait_mode(self, mode, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.robot_mode() == mode:
                return
            time.sleep(0.1)
        raise NovaError(f"timeout waiting for RobotMode {mode}")

    # ------------------------------------------------------------ joints
    def get_servo_angles(self):
        """Raw joint angles as the robot reports them [deg]."""
        return np.array([float(v) for v in self._send("GetAngle()")[:6]])

    def get_q(self):
        """Current joint angles in YOUR virtual-zero frame [rad]."""
        return self.map.servo_to_visual(self.get_servo_angles())

    def _check_limits(self, servo_deg):
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        bad = np.where((servo_deg < lo) | (servo_deg > hi))[0]
        if bad.size:
            msg = ", ".join(f"J{i + 1}={servo_deg[i]:.1f} (limit {lo[i]:.0f}..{hi[i]:.0f})"
                            for i in bad)
            raise NovaError(f"target outside servo limits: {msg}")

    # ------------------------------------------------------------ motion
    def move_to_q(self, q_rad, speed=10, accel=10, wait=True, tol_deg=0.3, timeout=60.0):
        """
        Joint move (MovJ) to q = [q1..q6] in radians, virtual-zero frame.
        speed / accel are percentages 1..100. Returns the servo angles sent.
        """
        servo = self.map.visual_to_servo(q_rad)
        self._check_limits(servo)
        j = ",".join(f"{a:.4f}" for a in servo)
        self._send(f"MovJ(joint={{{j}}},a={int(accel)},v={int(speed)})")
        if wait:
            self.wait_reached(servo, tol_deg, timeout)
        return servo

    def move_home(self, **kw):
        """Go to visual zero (= your servo home pose)."""
        return self.move_to_q(np.zeros(6), **kw)

    def servo_j(self, q_rad, t=0.1, aheadtime=50, gain=500):
        """
        Streaming target (ServoJ), for online control. Call every ~30 ms.
        Not affected by global speed, so send smooth, speed-planned points.
        """
        servo = self.map.visual_to_servo(q_rad)
        self._check_limits(servo)
        j = ",".join(f"{a:.4f}" for a in servo)
        self._send(f"ServoJ({j},t={t},aheadtime={aheadtime},gain={gain})")

    def wait_reached(self, servo_target, tol_deg=0.3, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            mode = self.robot_mode()
            if mode in (MODE_ERROR, MODE_COLLISION):
                raise NovaError(f"robot entered mode {mode} (error/collision) during move")
            if np.all(np.abs(self.get_servo_angles() - servo_target) < tol_deg):
                return
            time.sleep(0.05)
        raise NovaError("timeout waiting for target")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.5.1")
    ap.add_argument("--move", action="store_true",
                    help="enable the robot and go to visual zero, then jog J6 +5 deg and back")
    args = ap.parse_args()

    with NovaClient(args.ip) as robot:
        robot.request_control()
        print("servo deg :", np.round(robot.get_servo_angles(), 2))
        print("visual deg:", np.round(np.degrees(robot.get_q()), 2))

        if args.move:
            robot.enable()
            robot.move_home(speed=10, accel=10)
            q = robot.get_q()
            q[5] += np.radians(5)
            robot.move_to_q(q, speed=5, accel=5)
            q[5] -= np.radians(5)
            robot.move_to_q(q, speed=5, accel=5)
            print("done, visual deg:", np.round(np.degrees(robot.get_q()), 2))
