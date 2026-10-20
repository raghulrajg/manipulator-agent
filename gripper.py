"""
gripper.py
-----------
The Feetech SMS/STS servo gripper on its USB-RS485 adapter, through the
scservo_sdk package (in this repo) -- the same hardware and settings as
x-trainer's DobotGripper (xtrainer_utils/dobot_control/gripper):

    left arm  (robot 192.168.5.1): servo ID 21 on /dev/ttyUSB1
    right arm (robot 192.168.5.2): servo ID 22 on /dev/ttyUSB0
    raw position 2048 open, 3052 closed, 1 Mbaud

Positions are 0 (open) .. 255 (closed) and speeds 1..100 %, scaled to the
servo's raw counts the way DobotGripper does. The torque limit (register
48, 0..1000, default 300 as in x-trainer) is what the fingers squeeze with.

close() drives towards fully closed, waits until the fingers stop and says
whether they stopped on something: fingers that end up (nearly) fully
closed had nothing between them.

Run (opens and closes the gripper once):
    python3 gripper.py --arm left open       # ID 21 on /dev/ttyUSB1: open it
    python3 gripper.py --arm left close
    python3 gripper.py --arm left            # test: open, close, open
    python3 gripper.py --port /dev/ttyUSB0 --id 22
"""

import argparse
import time
from dataclasses import dataclass

TORQUE_LIMIT_ADDR = 48   # SMS/STS torque limit, 2 bytes, 0..1000
PRESENT_LOAD_ADDR = 60   # SMS/STS present load, 2 bytes, bit 10 = direction

# x-trainer: dobot_settings.ini GRIPPER_LEFT / GRIPPER_RIGHT, paired with the
# robot IPs in dobot_control/robots/dobot.py
LEFT = dict(port="/dev/ttyUSB1", servo_id=21, robot_ip="192.168.5.1")
RIGHT = dict(port="/dev/ttyUSB0", servo_id=22, robot_ip="192.168.5.2")


class GripperError(RuntimeError):
    pass


@dataclass
class GripResult:
    holding: bool    # the fingers stopped on something
    position: int    # 0 open .. 255 closed
    raw: int         # servo position counts
    load: int        # present load, 0.1 % of max, signed


class FeetechGripper:
    def __init__(self, port="/dev/ttyUSB1", servo_id=21, open_raw=2048, closed_raw=3052,
                 torque_limit=300, baudrate=1_000_000, empty_margin_raw=30, connect=True):
        """
        empty_margin_raw  fingers that stop closer than this to closed_raw
                          [servo counts] had nothing between them
        """
        self.port, self.servo_id = port, servo_id
        self.open_raw, self.closed_raw = int(open_raw), int(closed_raw)
        self.torque_limit, self.baudrate = int(torque_limit), int(baudrate)
        self.empty_margin_raw = int(empty_margin_raw)
        self.port_handler = self.servo = None
        if connect:
            self.connect()

    # ------------------------------------------------------------ connection
    def connect(self):
        from scservo_sdk import PortHandler, sms_sts   # needs pyserial
        self.port_handler = PortHandler(self.port)
        if not self.port_handler.setBaudRate(self.baudrate):   # also opens the port
            raise GripperError(f"cannot open {self.port} at {self.baudrate} baud")
        self.servo = sms_sts(self.port_handler)
        try:
            _, comm, err = self.servo.ping(self.servo_id)
            self._check(comm, err, f"ping servo {self.servo_id} on {self.port}")
            comm, err = self.servo.write2ByteTxRx(self.servo_id, TORQUE_LIMIT_ADDR, self.torque_limit)
            self._check(comm, err, "set the torque limit")
        except Exception:
            self.disconnect()
            raise
        return self

    def disconnect(self):
        if self.port_handler is not None:
            self.port_handler.closePort()
        self.port_handler = self.servo = None

    def _check(self, comm, err, what):
        if comm != 0:   # COMM_SUCCESS
            raise GripperError(f"{what}: {self.servo.getTxRxResult(comm)}")
        if err != 0:
            raise GripperError(f"{what}: {self.servo.getRxPacketError(err)}")

    # ------------------------------------------------------------ raw I/O
    def to_raw(self, position):
        return int(round(self.open_raw + position / 255.0 * (self.closed_raw - self.open_raw)))

    def from_raw(self, raw):
        return int(round((raw - self.open_raw) / (self.closed_raw - self.open_raw) * 255.0))

    def read_raw(self):
        raw, comm, err = self.servo.ReadPos(self.servo_id)
        self._check(comm, err, "read the position")
        return int(raw)

    def read_load(self):
        load, comm, err = self.servo.read2ByteTxRx(self.servo_id, PRESENT_LOAD_ADDR)
        self._check(comm, err, "read the load")
        return int(self.servo.scs_tohost(load, 10))

    def is_moving(self):
        moving, comm, err = self.servo.ReadMoving(self.servo_id)
        self._check(comm, err, "read the moving flag")
        return bool(moving)

    # ------------------------------------------------------------ motion
    def move(self, position, speed=100):
        """Go to position 0 (open) .. 255 (closed) at speed 1..100 %."""
        if not 0 <= position <= 255:
            raise ValueError("position must be 0..255")
        if not 1 <= speed <= 100:
            raise ValueError("speed must be 1..100")
        comm, err = self.servo.WritePosEx(self.servo_id, self.to_raw(position), int(speed / 100 * 4096), 0)
        self._check(comm, err, f"move to {position}")

    def wait_stopped(self, timeout=3.0, settle_s=0.2, still_raw=3):
        """Wait until the fingers have moved less than still_raw counts for
        settle_s (stalled on an object or at the goal); return the raw
        position."""
        t0 = time.monotonic()
        time.sleep(0.05)   # let it start
        last, since = self.read_raw(), time.monotonic()
        while time.monotonic() - t0 < timeout:
            time.sleep(0.02)
            raw = self.read_raw()
            if abs(raw - last) > still_raw:
                last, since = raw, time.monotonic()
            elif time.monotonic() - since >= settle_s and (not self.is_moving() or
                                                          time.monotonic() - since >= 2 * settle_s):
                return raw
        raise GripperError(f"gripper still moving after {timeout:.1f} s")

    def open(self, speed=100, wait=True):
        self.move(0, speed)
        if wait:
            raw = self.wait_stopped()
            if abs(raw - self.open_raw) > 3 * self.empty_margin_raw:
                raise GripperError(f"gripper did not open (raw {raw}, open is {self.open_raw})")

    def close(self, speed=50):
        """Close until the fingers stop; GripResult.holding is False if they
        closed (nearly) all the way, i.e. on nothing."""
        self.move(255, speed)
        raw = self.wait_stopped()
        return GripResult(holding=abs(self.closed_raw - raw) > self.empty_margin_raw,
                          position=self.from_raw(raw), raw=raw, load=self.read_load())


class FakeGripper:
    """Same interface, no hardware: closes onto object_position (0..255)
    if there is an object, else all the way. on_close(), if given, is
    called when it closes (the tests record the arm's joints there)."""

    def __init__(self, object_position=140, on_close=None):
        self.object_position = object_position
        self.on_close = on_close
        self.position = 0
        self.events = []

    def open(self, speed=100, wait=True):
        self.position = 0
        self.events.append("open")

    def close(self, speed=50):
        if self.on_close:
            self.on_close()
        holding = self.object_position is not None
        self.position = self.object_position if holding else 255
        self.events.append("close")
        return GripResult(holding=holding, position=self.position, raw=2048 + self.position * 4,
                          load=250 if holding else 20)

    def disconnect(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="Open or close the Feetech gripper, or test it (open, close, open).")
    parser.add_argument("action", nargs="?", choices=("open", "close", "test"), default="test")
    parser.add_argument("--arm", choices=("left", "right"), default="left",
                        help="left = ID 21 on /dev/ttyUSB1 (default), right = ID 22 on /dev/ttyUSB0")
    parser.add_argument("--port", help="override the arm's port")
    parser.add_argument("--id", type=int, help="override the arm's servo ID")
    parser.add_argument("--torque", type=int, default=300, help="torque limit 0..1000")
    args = parser.parse_args()

    side = LEFT if args.arm == "left" else RIGHT
    g = FeetechGripper(args.port or side["port"], side["servo_id"] if args.id is None else args.id,
                       torque_limit=args.torque)
    print(f"gripper: servo ID {g.servo_id} on {g.port}")
    try:
        print(f"now:   raw {g.read_raw()} (position {g.from_raw(g.read_raw())}/255)")
        if args.action in ("open", "test"):
            g.open()
            print(f"open:  raw {g.read_raw()}")
        if args.action in ("close", "test"):
            r = g.close()
            print(f"close: raw {r.raw} (position {r.position}/255), load {r.load}, "
                  f"{'holding something' if r.holding else 'nothing between the fingers'}")
        if args.action == "test":
            g.open()
            print(f"open:  raw {g.read_raw()}")
    finally:
        g.disconnect()


if __name__ == "__main__":
    main()
