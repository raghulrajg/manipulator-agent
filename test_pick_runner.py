"""
test_pick_runner.py
--------------------
Checks the real-time side with no robot, gripper, camera or broker:

  - dobot_tcp.py against fake_dobot.py (a fake Nova 2 controller on
    localhost): replies, error codes, waiting for a move, collision, a move
    that stops short, firmware without GetCurrentCommandID, joint limits
  - gripper.py against a fake Feetech servo: position scaling, a grip on
    an object, closing on nothing
  - mqtt_link.py: the messages, and PoseGate (ready -> found -> still 2.5 s
    -> sent once)
  - pick_runner.py end to end: ready -> pose -> plan -> MovJ / MovL -> the
    gripper closes only at the grasp -> lift -> place -> home; a missed grip;
    an unreachable object; the arm moving while the camera looks; a wrong
    kinematic model
  - the command line: options that would be unsafe on the robot are
    refused, failed cycles give exit code 1, an unreachable place spot is named

Run: python3 test_pick_runner.py
     python3 test_pick_runner.py --broker localhost    # also through a real MQTT broker
"""

import argparse
import json
import socket
import threading
import time

import numpy as np

import manipulator_kinematics as _mk

_mk.set_camera_calibration(None)   # the tests check the camera table, not a saved calibration

from dobot_tcp import DobotError, DobotTCP, error_text
from fake_dobot import FakeDobot
from gripper import FakeGripper, FeetechGripper
from manipulator_kinematics import IKError, fk, joint_map, pose
from mqtt_link import (
    POSE_TOPIC, READY_TOPIC, STATUS_TOPIC, LocalBus, MessageError, PoseGate,
    parse_pose_message, parse_ready, pose_message, ready_message,
)
from pick_runner import DEMO_T_CAMERA_OBJECT, PickFailed, PickRunner, start_fake_vision

HOME = joint_map.home_servo_deg


def connected(fake):
    robot = DobotTCP("127.0.0.1", fake.port, poll_s=0.01).connect()
    robot.request_control()
    robot.enable()
    return robot


def names(fake):
    return [n for n, _, _ in fake.log if n in ("MovJ", "MovL")]


# ============================================================
# DOBOT TCP
# ============================================================

def test_dobot_client():
    print("=" * 60)
    print("dobot_tcp.py against the fake controller")
    print("=" * 60)
    fake = FakeDobot().start()
    robot = connected(fake)
    assert robot.robot_mode() == 5
    assert np.allclose(robot.get_angle(), HOME)
    flange = fk(np.zeros(6), "flange")[:3, 3] * 1000
    assert np.allclose(robot.get_pose()[:3], flange, atol=1e-3)

    target = np.array([-150.0, 10.0, -80.0, 5.0, 90.0, 120.0])
    t0 = time.monotonic()
    q = robot.move_j(target, v=100, tol_deg=0.01)
    assert np.allclose(q, target, atol=0.01) and time.monotonic() - t0 > 0.15, "returned before the move ended"

    # GetCurrentCommandID tells this move from the last one: no early return
    back = robot.mov_j(HOME, v=100)
    assert back == 2, back
    q = robot.wait_motion(back, lambda q: True)
    assert np.allclose(q, HOME, atol=1e-6), "wait_motion returned while the arm was still moving"

    sent = len(fake.log)
    for bad in ([0, 170, 0, 0, 0, 0], [0, 0, 0, 0, 0, float("nan")]):
        try:
            robot.mov_j(bad)
            raise AssertionError("sent a target outside the joint limits")
        except DobotError as e:
            assert "joint limits" in str(e)
    assert len(fake.log) == sent, "an out-of-limit target reached the controller"

    try:
        robot.send("Nope()")
        raise AssertionError("ErrorID -10000 not raised")
    except DobotError as e:
        assert e.error_id == -10000 and "does not exist" in str(e)
    assert "optional parameter range" in error_text(-60002)
    robot.close()
    fake.stop()
    print("OK: replies parsed, moves waited for to the end, limits checked before sending\n")


def test_dobot_faults():
    print("=" * 60)
    print("Collision, a move that stops short, no GetCurrentCommandID")
    print("=" * 60)
    target = np.array([-150.0, 10.0, -80.0, 5.0, 90.0, 120.0])

    fake = FakeDobot(fault=("collision", 1)).start()
    robot = connected(fake)
    try:
        robot.move_j(target, v=100)
        raise AssertionError("collision not reported")
    except DobotError as e:
        assert "collision" in str(e) and "[22]" in str(e), e
    fake.stop()

    fake = FakeDobot(fault=("short", 1)).start()
    robot = connected(fake)
    t0 = time.monotonic()
    try:
        robot.move_j(target, v=100, timeout=20)
        raise AssertionError("stopping 1 deg short not reported")
    except DobotError as e:
        assert "not at the target" in str(e) and time.monotonic() - t0 < 5, e
    fake.stop()

    # firmware without GetCurrentCommandID: idle + at the target is enough
    fake = FakeDobot(command_id=False).start()
    robot = connected(fake)
    assert np.allclose(robot.move_j(target, v=100), target, atol=0.3)
    assert robot.current_command_id() is None
    fake.stop()

    # the timeout is for a robot that stopped moving, not for a slow move
    fake = FakeDobot(deg_per_s=20).start()   # this move takes 1.5 s
    robot = connected(fake)
    assert np.allclose(robot.move_j(target, v=100, timeout=0.5), target, atol=0.3)
    fake.stop()
    fake = FakeDobot(command_id=False, fault=("short", 1)).start()
    robot = connected(fake)
    t0 = time.monotonic()
    try:
        robot.move_j(target, v=100, timeout=1.0)
        raise AssertionError("stopping short without GetCurrentCommandID not reported")
    except DobotError as e:
        assert "no joint has moved" in str(e) and time.monotonic() - t0 < 5, e
    fake.stop()

    # stuck "running" with readings jittering 0.02 deg: noise is not progress
    fake = FakeDobot(fault=("stuck", 1)).start()
    robot = connected(fake)
    t0 = time.monotonic()
    try:
        robot.move_j(target, v=100, timeout=1.0)
        raise AssertionError("a stuck robot with jittering readings was waited for forever")
    except DobotError as e:
        assert "no joint has moved" in str(e) and time.monotonic() - t0 < 5, e
    robot.stop()
    fake.stop()

    # the fullwidth semicolon some firmware sends
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def reply():
        conn, _ = server.accept()
        conn.recv(100)
        conn.sendall("0,{5},RobotMode()；".encode())
        conn.close()

    threading.Thread(target=reply, daemon=True).start()
    robot = DobotTCP("127.0.0.1", server.getsockname()[1]).connect()
    assert robot.robot_mode() == 5
    server.close()
    print("OK\n")


# ============================================================
# GRIPPER
# ============================================================

class FakeServo:
    """Feetech servo: moves 40 counts per read towards the goal, stops at
    block_raw if an object is in the way. Closing is towards lower counts
    (the gripper's defaults: 3052 open, 2048 closed)."""

    def __init__(self, raw=3052, block_raw=None):
        self.raw, self.goal, self.block_raw = raw, raw, block_raw

    def WritePosEx(self, scs_id, position, speed, acc):
        self.goal = position
        return 0, 0

    def ReadPos(self, scs_id):
        step = int(np.clip(self.goal - self.raw, -40, 40))
        new = self.raw + step
        if self.block_raw is not None and step < 0:
            new = max(new, self.block_raw)
        self.raw = new
        return self.raw, 0, 0

    def ReadMoving(self, scs_id):
        return int(self.raw != self.goal and not (self.block_raw is not None and self.raw <= self.block_raw)), 0, 0

    def read2ByteTxRx(self, scs_id, address):
        return (1024 | 300) if self.raw == self.block_raw else 15, 0, 0   # bit 10 = direction

    def scs_tohost(self, a, b):
        return -(a & ~(1 << b)) if a & (1 << b) else a


def test_gripper():
    print("=" * 60)
    print("gripper.py with a fake Feetech servo")
    print("=" * 60)
    g = FeetechGripper(connect=False)
    assert g.to_raw(0) == 3052 and g.to_raw(255) == 2048 and g.from_raw(2048) == 255
    g.servo = FakeServo(block_raw=2600)
    r = g.close()
    assert r.holding and r.raw == 2600 and r.load == -300, r
    g.open()
    assert g.servo.raw == 3052
    g.servo = FakeServo()
    r = g.close()
    assert not r.holding and r.raw == 2048 and r.position == 255, r
    for bad in (-1, 256):
        try:
            g.move(bad)
            raise AssertionError("position out of range accepted")
        except ValueError:
            pass
    print(f"OK: grip on an object stops at raw 2600 (holding), on nothing at {r.raw} (not holding)\n")


# ============================================================
# MQTT MESSAGES AND THE VISION GATE
# ============================================================

def test_messages_and_gate():
    print("=" * 60)
    print("mqtt_link.py messages and PoseGate")
    print("=" * 60)
    T = pose(np.eye(3), [0.03, -0.02, 0.38])
    msg = parse_pose_message(json.dumps(pose_message(T, "abc", "tomato", "center", 0.06, 2.5, True)).encode())
    assert np.allclose(msg["T_camera_object"], T) and msg["request_id"] == "abc" and msg["symmetric"]
    assert msg["diameter"] == 0.06 and msg["origin"] == "center"
    T_rounded = np.round(pose(np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0.0], [-0.48, 0.64, 0.6]]), [0, 0, 0.3]), 4)
    R = parse_pose_message(json.dumps({"T_camera_object": T_rounded.tolist()}))["T_camera_object"][:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12), "rotation not snapped"
    for bad in ({"T_camera_object": np.eye(3).tolist()}, {"T_camera_object": np.diag([1, 1, -1, 1]).tolist()},
                {"T_camera_object": (np.eye(4) * 2).tolist()}, {"pose": 1}, "not json"):
        try:
            parse_pose_message(bad if isinstance(bad, str) else json.dumps(bad))
            raise AssertionError(f"accepted {bad}")
        except MessageError:
            pass
    assert parse_ready(b"ready")["request_id"] is None
    for bad_id in (["x"], {"id": 1}, True, 1.5):
        for parse, msg in ((parse_ready, {"request_id": bad_id}),
                           (parse_pose_message, {"request_id": bad_id, "T_camera_object": T.tolist()})):
            try:
                parse(json.dumps(msg))
                raise AssertionError(f"{parse.__name__} accepted request_id {bad_id!r}")
            except MessageError:
                pass
    assert parse_ready(json.dumps(ready_message("r1", HOME)))["request_id"] == "r1"

    gate = PoseGate(settle_s=2.5, max_move=0.01)
    assert gate.update(T, 0.0) is None, "sent without a ready"
    assert gate.arm("r1")
    assert gate.update(T, 0.0) is None and gate.update(T, 2.4) is None
    assert gate.update(None, 2.45) is None, "lost"                              # restarts the wait
    assert gate.update(T, 2.5) is None and gate.update(T, 4.9) is None
    moved = pose(np.eye(3), [0.05, -0.02, 0.38])
    assert gate.update(moved, 5.0) is None, "moved 2 cm"                         # restarts the wait
    assert gate.update(moved, 7.4) is None
    sent = gate.update(moved, 7.5)
    assert sent is not None and np.allclose(sent, moved), "not sent after 2.5 s still"
    assert gate.update(moved, 8.0) is None, "sent twice"
    assert not gate.arm("r1"), "a repeated ready re-armed the gate"
    assert gate.arm("r2") and gate.update(T, 10.0) is None and gate.update(T, 12.5) is not None
    print("OK: sent once, only after a ready and 2.5 s of steady tracking\n")


# ============================================================
# PICK RUNNER END TO END
# ============================================================

def run_pick(T_camera_object=DEMO_T_CAMERA_OBJECT, place=(-0.30, 0.30), object_position=140,
             fault=None, nudge=False, table_z=None):
    """One cycle on the fake robot / gripper / bus. Returns (runner, fake,
    gripper, bus, result or exception, joints when the gripper closed)."""
    fake = FakeDobot(fault=fault).start()
    robot = DobotTCP("127.0.0.1", fake.port, poll_s=0.01)
    closed_at = []
    gripper = FakeGripper(object_position, on_close=lambda: closed_at.append(fake.joints.copy()))
    bus = LocalBus()
    if nudge:   # someone jogs the arm while the camera looks
        ready_q = bus.subscribe(READY_TOPIC)

        def vision():
            ready = parse_ready(ready_q.get())
            fake.joints = fake.joints + 0.5
            bus.publish(POSE_TOPIC, pose_message(T_camera_object, ready["request_id"], diameter=0.06))

        threading.Thread(target=vision, daemon=True).start()
    else:
        start_fake_vision(bus, T_camera_object, delay=0.05)
    log = []
    runner = PickRunner(robot.connect(), gripper, bus, table_z=table_z, place_xy=place, pose_timeout=10,
                        speed=100, line_speed=100, log=log.append)
    try:
        runner.startup()
        result = runner.cycle()
    except (PickFailed, IKError, DobotError) as e:
        result = e
    robot.close()
    fake.stop()
    return runner, fake, gripper, bus, result, closed_at, log


def test_pick_and_place():
    print("=" * 60)
    print("pick_runner.py: ready -> pose -> pick -> place -> home")
    print("=" * 60)
    runner, fake, gripper, bus, plan, closed_at, log = run_pick()
    assert not isinstance(plan, Exception), plan
    assert names(fake) == ["MovJ", "MovL", "MovL", "MovJ", "MovL", "MovL", "MovJ"], names(fake)
    assert gripper.events == ["open", "open", "close", "open"], gripper.events

    # the gripper closed only once the arm was at the grasp, tool tip on the object's center
    grasp = next(w for w in plan.waypoints if w.name == "grasp")
    assert len(closed_at) == 1 and np.abs(closed_at[0] - grasp.q_servo_deg).max() < 0.3, closed_at
    tip = fk(joint_map.servo_to_visual(closed_at[0]), "tool")[:3, 3]
    assert np.linalg.norm(tip - plan.T_base_object[:3, 3]) < 0.002, tip
    assert abs(plan.table_z - (plan.T_base_object[2, 3] - 0.03)) < 1e-9, "table = center - diameter/2"

    # every MovL is a straight vertical line (same x y, same joints-to-pose branch)
    movl = [np.array([float(v) for v in a.split("{")[1].split("}")[0].split(",")]) for n, a, _ in fake.log if n == "MovL"]
    start = [j for n, _, j in fake.log if n == "MovL"]
    for a, b in zip(start, movl):
        pa, pb = (fk(joint_map.servo_to_visual(q), "tool")[:3, 3] for q in (a, b))
        assert np.linalg.norm(pa[:2] - pb[:2]) < 0.002 and abs(abs(pa[2] - pb[2]) - 0.10) < 0.002, (pa, pb)

    # put down at the place spot, at the height it was picked from
    put = fk(joint_map.servo_to_visual(movl[2]), "tool")[:3, 3]   # grasp, lift, place, retreat
    assert np.allclose(put, [-0.30, 0.30, plan.T_base_object[2, 3]], atol=0.002), put

    assert np.allclose(fake.joints, HOME, atol=0.3), "not back home"
    states =[m["state"] for t, m in bus.sent if t == STATUS_TOPIC]
    assert states == ["waiting", "picking", "holding", "placed", "done"], states
    ready = [m for t, m in bus.sent if t == READY_TOPIC]
    assert ready[0]["request_id"] == runner.request_id and np.allclose(ready[0]["joints"], HOME)
    print(f"OK: {len(names(fake))} moves, gripper closed at the grasp "
          f"({np.linalg.norm(tip - plan.T_base_object[:3, 3]) * 1000:.2f} mm from the object), back home\n")


def test_pick_failures():
    print("=" * 60)
    print("pick_runner.py: missed grip, unreachable object, arm moved, bad model")
    print("=" * 60)
    _, fake, gripper, bus, result, _, _ = run_pick(object_position=None)
    assert isinstance(result, PickFailed) and "nothing" in str(result), result
    assert names(fake) == ["MovJ", "MovL", "MovL", "MovJ"], names(fake)   # down, up, home: no place
    assert gripper.events[-2:] == ["close", "open"] and np.allclose(fake.joints, HOME, atol=0.3)

    far = pose(np.eye(3), [0.9, 0.0, 0.38])
    _, fake, gripper, _, result, _, _ = run_pick(far)
    assert isinstance(result, IKError) and names(fake) == [], (result, names(fake))

    _, fake, gripper, _, result, _, _ = run_pick(nudge=True)
    assert isinstance(result, DobotError) and "moved" in str(result) and names(fake) == [], result

    # controller's base frame turned 180 deg: refuse before moving
    fake = FakeDobot().start()
    fake._run_orig = fake._run

    def turned(name, args):
        err, values = fake._run_orig(name, args)
        if name == "GetPose":
            v = [float(x) for x in values.split(",")]
            values = ",".join(str(x) for x in [-v[0], -v[1], *v[2:]])
        return err, values

    fake._run = turned
    robot = DobotTCP("127.0.0.1", fake.port, poll_s=0.01).connect()
    try:
        PickRunner(robot, FakeGripper(), LocalBus(), log=lambda s: None).startup()
        raise AssertionError("ran with a model 700 mm off")
    except DobotError as e:
        assert "kinematic model" in str(e), e
    fake.stop()

    # the collision during the descent stops it with the error
    _, fake, gripper, _, result, closed_at, _ = run_pick(fault=("collision", 2))
    assert isinstance(result, DobotError) and "collision" in str(result) and not closed_at, result
    print("OK: missed grip goes back up and home; unreachable, moved arm and wrong model never move; "
          "collision stops it\n")


def test_command_line():
    print("=" * 60)
    print("pick_runner.py command line: refusals, exit code, place spot")
    print("=" * 60)
    import os
    import subprocess
    import sys
    import tempfile

    def run_cli(*args):
        return subprocess.run([sys.executable, "pick_runner.py", *args], capture_output=True, text=True, timeout=60)

    for args, why in ((["--robot-ip", "192.0.2.1"], "--table-z is required"),
                      (["--table-z", "-0.05", "--fake-vision", "demo"], "with --dry-run"),
                      (["--table-z", "-47"], "it is in meters"),
                      (["--table-z", "0.018", "--robot-ip", "192.0.2.1"], "pass --arm"),
                      (["--table-z", "-0.10", "--robot-ip", "192.0.2.1"], "--pedestal"),
                      (["--table-z", "-0.05", "--cycles", "0"], "--cycles 1"),
                      (["--dry-run", "--speed", "0"], "1..100")):
        r = run_cli(*args)
        assert r.returncode == 2 and why in r.stderr, (args, r.returncode, r.stderr[-300:])

    with tempfile.TemporaryDirectory() as tmp:
        far = os.path.join(tmp, "far.txt")
        np.savetxt(far, pose(np.eye(3), [0.9, 0.0, 0.38]))
        r = run_cli("--dry-run", "--fake-vision", far, "--table-z", "-0.05")
        assert r.returncode == 1 and "[failed]" in r.stdout and "1 cycles, 1 failed" in r.stdout, r.stdout[-500:]
    r = run_cli("--dry-run", "--fake-vision", "demo", "--table-z", "-0.05", "--place", "0", "0")
    assert r.returncode == 1 and "place spot [0.0, 0.0]" in r.stdout, r.stdout[-500:]
    r = run_cli("--dry-run", "--fake-vision", "demo", "--place", "-0.30", "0.30", "--cycles", "2", "--speed-factor", "50")
    assert r.returncode == 0 and r.stdout.count("[done]") == 2 and "2 cycles, 0 failed" in r.stdout, r.stdout[-500:]
    print("OK: bad options refused before connecting; a failed cycle exits 1; the place spot is named\n")


def test_real_broker(host):
    print("=" * 60)
    print(f"MqttLink through the broker at {host}")
    print("=" * 60)
    from mqtt_link import MqttLink
    a = MqttLink("test_a", host=host)
    b = MqttLink("test_b", host=host)
    got = a.subscribe(READY_TOPIC + "/test")
    a.connect()
    b.connect()
    time.sleep(0.3)
    b.publish(READY_TOPIC + "/test", ready_message("x1"))
    assert parse_ready(got.get(timeout=5))["request_id"] == "x1"
    a.close()
    b.close()
    print("OK\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", help="also test MqttLink through this MQTT broker")
    args = parser.parse_args()
    test_dobot_client()
    test_dobot_faults()
    test_gripper()
    test_messages_and_gate()
    test_pick_and_place()
    test_pick_failures()
    test_command_line()
    if args.broker:
        test_real_broker(args.broker)
