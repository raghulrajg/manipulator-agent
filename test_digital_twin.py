"""
test_digital_twin.py
---------------------
Checks the digital twin with no robot, gripper or camera, against
fake_dobot.py (its dashboard and real-time feedback ports):

  - dobot_feedback.py: the 1440-byte packet at the guide's byte offsets,
    bad packets refused, a stream that starts mid-packet re-synchronised
  - the feedback follows the robot at its 8 ms rate, including moves made
    from the pendant or by hand in drag mode
  - Dobot's URDF chain (what the browser draws) matches manipulator_kinematics
  - robot -> twin: whatever moves the arm, the twin shows it
  - twin -> robot: joint moves, straight tool jogs (checked along the line),
    pose moves, the gripper, pick and place, Detect with the camera
  - refused: disarmed, robot not enabled, joint limits, below the table,
    busy, bad sizes; STOP halts a move and disarms; drag mode disarms
  - the WebSocket protocol, and a robot that stops when the last browser
    goes away
  - a real browser (if Playwright is installed): the page loads the Nova 2
    meshes, mirrors the robot, and moves it from its buttons

Run: python3 test_digital_twin.py
"""

import asyncio
import json
import math
import os
import socket
import struct
import tempfile
import threading
import time

import numpy as np

import manipulator_kinematics as _mk

_mk.set_camera_calibration(None)   # the tests check the camera table, not a saved calibration

from digital_twin import WEB, Twin, demo_pose_on_table, load_model, make_app, model_fk, T_LINK6_FLANGE
from dobot_feedback import FIELDS, PACKET_SIZE, FeedbackError, FeedbackReader, pack_feedback, parse_feedback
from dobot_tcp import DobotError, DobotTCP
from fake_dobot import FakeDobot
from gripper import FakeGripper
from manipulator_kinematics import SERVO_LIMITS_DEG, TableConstraint, fk, joint_map
from mqtt_link import POSE_TOPIC, READY_TOPIC, LocalBus
from pick_runner import start_fake_vision

HOME = joint_map.home_servo_deg
TOKEN = "test-token-123"
CONFIG = {"sim": True, "robot": None, "table_z": 0.0, "detect": False, "max_speed": 100, "gripper": "simulated"}


def wait_for(cond, timeout=10.0, what="condition"):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


class Harness:
    """A twin on a fake robot, in this process."""

    def __init__(self, table_z=0.0, place=None, fake_vision=False, gripper=None, sim=True, **fake_kw):
        self.fake = FakeDobot(enabled=fake_kw.pop("enabled", True), **fake_kw).start()
        self.robot = DobotTCP("127.0.0.1", self.fake.port, poll_s=0.01, motion_port=self.fake.motion_port or 30003,
                              protocol=self.fake.protocol).connect()
        self.feedback = FeedbackReader("127.0.0.1", self.fake.feedback_port).start()
        self.feedback.wait_first()
        link = None
        if fake_vision:
            link = LocalBus()
            start_fake_vision(link, demo_pose_on_table(table_z), delay=0.05)
        self.lines = []
        self.gripper = gripper or FakeGripper(object_position=None)
        self.twin = Twin(self.robot, self.feedback, self.gripper, link, table_z=table_z,
                         max_speed=100, place_xy=place, sim=sim, log=self.lines.append)
        self.twin.fake = self.fake
        self.twin.check_model()
        self.twin.cmd_speed(100)
        self.twin.cmd_set_object(xy=[-0.42, 0.12])   # as --sim starts

    def ok(self, cmd, **args):
        r = self.twin.handle({"cmd": cmd, **args})
        assert r["ok"], (cmd, r)
        return r

    def refused(self, cmd, why, **args):
        r = self.twin.handle({"cmd": cmd, **args})
        assert not r["ok"] and why in r["error"], (cmd, why, r)
        return r["error"]

    def idle(self, timeout=60):
        wait_for(lambda: self.twin.task is None, timeout, "the twin's task to end")
        # and the feedback (what the browser sees) has caught up with the controller
        wait_for(lambda: np.allclose(self.feedback.latest()["q_actual"], self.robot.get_angle(), atol=1e-3),
                 2, "the feedback to settle")

    def motions_after(self, name):
        """MovJ / MovL the controller received after the last `name` command."""
        names = [n for n, _, _ in self.fake.log]
        last = max(i for i, n in enumerate(names) if n == name)
        return [n for n in names[last + 1:] if n in ("MovJ", "MovL")]

    def joints(self):
        return np.array(self.twin.snapshot()["robot"]["joints"])

    def close(self):
        self.feedback.stop()
        self.robot.close()
        self.fake.stop()


# ============================================================
# FEEDBACK PORT
# ============================================================

def test_feedback_packets():
    print("=" * 60)
    print("Real-time feedback packet (port 30004 layout)")
    print("=" * 60)
    q = [-180.0, 1.5, -90.25, 3.0, 90.0, 91.5]
    tool = [-345.0, 117.5, 415.4, 180.0, 0.0, -90.0]
    pkt = pack_feedback(robot_mode=7, q_actual=q, tool_vector_actual=tool, current_command_id=42, drag_status=1)
    assert len(pkt) == PACKET_SIZE == 1440
    # byte offsets straight from the guide's table
    assert struct.unpack_from("<H", pkt, 0)[0] == 1440
    assert pkt[48:56] == bytes.fromhex("efcdab8967452301"), "test value 0x0123456789ABCDEF, little-endian"
    assert struct.unpack_from("<Q", pkt, 24)[0] == 7
    assert struct.unpack_from("<6d", pkt, 432) == tuple(q)
    assert struct.unpack_from("<6d", pkt, 624) == tuple(tool)
    assert pkt[1027] == 1 and pkt[1031] == 101 and struct.unpack_from("<Q", pkt, 1112)[0] == 42
    s = parse_feedback(pkt)
    assert s["robot_mode"] == 7 and np.allclose(s["q_actual"], q) and s["current_command_id"] == 42
    assert set(FIELDS) <= set(s)
    for bad in (pkt[:-1], b"\0" * 1440, pkt[:48] + struct.pack("<Q", 1) + pkt[56:]):
        try:
            parse_feedback(bad)
            raise AssertionError("bad packet accepted")
        except FeedbackError:
            pass

    # a stream joined in the middle of a packet: the reader finds the next one
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def stream():
        conn, _ = server.accept()
        with conn:
            conn.sendall(pkt[700:])            # the tail of a packet first
            for k in range(30):
                conn.sendall(pack_feedback(robot_mode=5, q_actual=[k, 0, 0, 0, 0, 0]))
                time.sleep(0.008)
            time.sleep(1)

    threading.Thread(target=stream, daemon=True).start()
    reader = FeedbackReader("127.0.0.1", server.getsockname()[1]).start()
    wait_for(lambda: (reader.latest() or {}).get("q_actual", [0])[0] == 29, 5, "the last packet")
    assert reader.resyncs >= 1 and reader.packets >= 20, (reader.resyncs, reader.packets)
    reader.stop()
    server.close()
    print(f"OK: offsets match the guide; bad packets refused; joined mid-packet, re-synced ({reader.resyncs}x)\n")


def test_feedback_follows_robot():
    print("=" * 60)
    print("Feedback follows the robot: own moves, pendant, drag mode")
    print("=" * 60)
    fake = FakeDobot(enabled=True).start()
    reader = FeedbackReader("127.0.0.1", fake.feedback_port).start()
    reader.wait_first()
    n0, t0 = reader.packets, time.monotonic()
    time.sleep(0.5)
    rate = (reader.packets - n0) / (time.monotonic() - t0)
    assert rate > 80, f"feedback at {rate:.0f} Hz, the controller sends 125 Hz"
    target = [-150.0, 10.0, -80.0, 5.0, 90.0, 120.0]
    fake.external_move(target, 0.3)   # pendant jog: no command from us
    time.sleep(0.05)
    assert np.allclose(reader.latest()["q_actual"], target, atol=1e-6)
    robot = DobotTCP("127.0.0.1", fake.port).connect()
    robot.start_drag()
    time.sleep(0.03)
    assert reader.latest()["robot_mode"] == 6 and reader.latest()["drag_status"] == 1
    fake.external_move(HOME, 0.3)     # moved by hand
    time.sleep(0.05)
    assert np.allclose(reader.latest()["q_actual"], HOME, atol=1e-6)
    robot.stop_drag()
    reader.stop()
    robot.close()
    fake.stop()
    print(f"OK: {rate:.0f} packets/s; pendant and hand moves show up at once\n")


def test_model():
    print("=" * 60)
    print("Dobot's Nova 2 URDF (what the browser draws) vs the DH kinematics")
    print("=" * 60)
    m = load_model()
    assert [j["name"] for j in m["joints"]] == [f"joint{i}" for i in range(1, 7)]
    for name, link in m["links"].items():
        if link["mesh"]:
            assert (WEB / link["mesh"].removeprefix("web/")).exists(), link["mesh"]
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(1000):
        s = rng.uniform(SERVO_LIMITS_DEG[:, 0], SERVO_LIMITS_DEG[:, 1])
        d = model_fk(m, s) @ T_LINK6_FLANGE - fk(joint_map.servo_to_visual(s), "flange")
        worst = max(worst, float(np.abs(d[:3, 3]).max()))
    assert worst < 1e-4, f"{worst * 1000:.3f} mm"
    tool = model_fk(m, HOME) @ np.array(m["link6_tool"])
    assert np.allclose(tool, fk(np.zeros(6), "tool"), atol=1e-4)
    print(f"OK: 7 meshes present; URDF chain = DH flange within {worst * 1000:.3f} mm; tool tip mounted right\n")


# ============================================================
# THE TWIN
# ============================================================

def below_table_target(table_z):
    """A joint target inside the limits whose MovJ path dips under the table."""
    rng = np.random.default_rng(11)
    table = TableConstraint(table_z)
    for _ in range(5000):
        q = HOME + rng.uniform(-40, 40, 6)
        if np.all((q > SERVO_LIMITS_DEG[:, 0]) & (q < SERVO_LIMITS_DEG[:, 1])) and \
                table.violations(joint_map.servo_to_visual(q)):
            return q
    raise AssertionError("no below-table target found")


def test_twin_refuses_unsafe():
    print("=" * 60)
    print("Twin refuses: disarmed, not enabled, limits, table, busy")
    print("=" * 60)
    h = Harness(enabled=False)
    h.refused("move_joints", "not armed", joints=HOME.tolist())
    h.refused("arm", "enable it first")
    h.ok("enable")
    h.ok("arm")
    h.refused("move_joints", "joint limits", joints=[-180, 160, -90, 0, 90, 90])
    h.refused("preview_joints", "joint limits", joints=[-180, 0, -170, 0, 90, 90])
    # a full J5 turn whose gripper dips between coarse samples (found by the review)
    A = np.array([-180, -16.47, -131.31, 40.54, -359.15, 90.0])
    B = A.copy()
    B[4] = 358.98
    try:
        h.twin.check_joint_move(A, B)
        raise AssertionError("a wrist sweep under the table passed the check")
    except Exception as e:
        assert "below the table" in str(e), e
    low = below_table_target(h.twin.table_z)
    h.refused("move_joints", "below the table", joints=low.tolist())
    h.refused("preview_joints", "below the table", joints=low.tolist())
    h.refused("jog_tool", "-100..100", axis="z", delta=-150)
    h.twin.table_z = 0.15      # tool tip 218 mm up: 100 mm lower would be under a table at 150 mm
    h.refused("jog_tool", "table", axis="z", delta=-100)
    h.twin.table_z = 0.0
    h.refused("speed", "1..100", value=0)
    h.refused("set_object", "diameter", xy=[-0.4, 0.1], diameter=0.5)
    sent = len([c for c in h.fake.log if c[0] in ("MovJ", "MovL")])
    assert sent == 0, "a refused move reached the robot"
    h.ok("speed", value=5)
    h.ok("move_joints", joints=(HOME + [60, 0, 0, 0, 0, 0]).tolist())
    h.refused("move_joints", "busy", joints=HOME.tolist())
    h.refused("drag", "busy", on=True)
    h.ok("stop")
    h.idle()
    assert not h.twin.armed and h.fake.mode == 5
    h.close()
    print("OK: every unsafe or out-of-turn command refused before reaching the robot\n")


def test_twin_two_way():
    print("=" * 60)
    print("Both ways: robot -> twin and twin -> robot")
    print("=" * 60)
    h = Harness()
    # robot -> twin: someone jogs it from the pendant
    q_pendant = HOME + [20, -10, 15, 0, -5, 30]
    h.fake.external_move(q_pendant, 0.4)
    wait_for(lambda: np.allclose(h.joints(), q_pendant, atol=1e-3), 1, "the twin to follow the pendant")
    # ... or by hand in drag mode (the twin disarms itself)
    h.ok("arm")
    h.ok("drag", on=True)
    assert not h.twin.armed
    wait_for(lambda: h.twin.snapshot()["robot"]["mode"] == 6, 1, "drag mode")
    h.fake.external_move(HOME, 0.4)
    wait_for(lambda: np.allclose(h.joints(), HOME, atol=1e-3), 1, "the twin to follow the hand")
    h.ok("drag", on=False)

    # twin -> robot: a joint move
    h.ok("arm")
    target = HOME + [-25, 10, -10, 5, 0, -20]
    h.ok("move_joints", joints=target.tolist())
    h.idle()
    assert np.allclose(h.fake.joints, target, atol=0.3) and np.allclose(h.joints(), target, atol=0.3)

    # a straight tool jog: every feedback sample on the way stays on the line
    h.ok("home")
    h.idle()
    start = fk(joint_map.servo_to_visual(h.robot.get_angle()), "tool")[:3, 3]
    samples = []
    h.ok("jog_tool", axis="z", delta=-40)
    while h.twin.task is not None:
        samples.append(fk(joint_map.servo_to_visual(h.feedback.latest()["q_actual"]), "tool")[:3, 3])
        time.sleep(0.01)
    end = fk(joint_map.servo_to_visual(h.joints()), "tool")[:3, 3]
    assert np.allclose(end, start + [0, 0, -0.04], atol=0.002), (start, end)
    off_line = max(float(np.linalg.norm(p[:2] - start[:2])) for p in samples)
    assert len(samples) > 5 and off_line < 0.001, (len(samples), off_line)

    # a pose target (as the draggable gizmo sends it)
    T = fk(joint_map.servo_to_visual(h.joints()), "tool")
    T[:3, 3] += [-0.05, 0.03, 0.02]
    h.ok("preview_pose", T=T.tolist())
    assert h.twin.snapshot()["ghost"] is not None
    h.ok("move_pose", T=T.tolist(), linear=False)
    h.idle()
    assert np.allclose(fk(joint_map.servo_to_visual(h.joints()), "tool")[:3, 3], T[:3, 3], atol=0.002)
    h.close()
    print(f"OK: pendant and hand moves mirrored; joint, line (max {off_line * 1000:.2f} mm off) and pose moves done\n")


def test_twin_stop():
    print("=" * 60)
    print("STOP halts a running move and disarms")
    print("=" * 60)
    h = Harness()
    h.ok("arm")
    h.ok("speed", value=5)
    target = HOME + [90, 0, 0, 0, 0, 0]
    h.ok("move_joints", joints=target.tolist())
    wait_for(lambda: h.fake.mode == 7, 2, "the move to start")
    time.sleep(1.5)
    h.ok("stop")
    h.idle(10)
    q = h.joints()
    assert not h.twin.armed and h.fake.mode == 5, q
    assert 2 < q[0] - HOME[0] < 80, f"J1 at {q[0]:.1f}: should have stopped part way"
    time.sleep(0.3)
    assert np.allclose(h.joints(), q, atol=1e-6), "still moving after STOP"
    assert any("STOP" in line for line in h.lines)
    h.close()
    print(f"OK: stopped at J1 {q[0]:.1f} (target {target[0]:.0f}), disarmed\n")


def test_twin_pick_and_place():
    print("=" * 60)
    print("Pick and place from the twin; a miss; pick only")
    print("=" * 60)
    h = Harness(place=[-0.25, 0.35])
    h.ok("set_object", xy=[-0.42, 0.12], diameter=0.06)
    plan = h.ok("plan")["plan"]
    assert [m["name"] for m in plan["moves"]] == ["pre_grasp", "grasp", "lift", "pre_place", "place", "retreat", "home"]
    assert len(plan["path"]) > 10
    h.ok("arm")
    h.ok("pick")
    h.idle(90)
    s = h.twin.snapshot()
    T = np.array(s["object"]["T"])
    assert np.allclose(T[:3, 3], [-0.25, 0.35, 0.03], atol=0.002) and not s["object"]["attached"], T[:3, 3]
    assert np.allclose(h.joints(), HOME, atol=0.3) and s["gripper"]["position"] == 0
    assert any("[placed]" in line for line in h.lines) and h.twin.armed

    # the fingers close on air (the object is not where the arm went): back up, home
    h.ok("set_object", xy=[-0.42, 0.12])
    h.twin._object_between_fingers = lambda: False
    h.ok("pick")
    h.idle(90)
    assert any("closed on nothing" in line for line in h.lines)
    assert np.allclose(h.joints(), HOME, atol=0.3)
    assert np.allclose(np.array(h.twin.snapshot()["object"]["T"])[:3, 3], [-0.42, 0.12, 0.03], atol=1e-9)
    del h.twin._object_between_fingers

    # pick only: it stays in the gripper at home, and follows the arm
    h.ok("set_place", xy=None)
    h.ok("pick")
    h.idle(90)
    s = h.twin.snapshot()
    assert s["object"]["attached"] and s["gripper"]["holding"]
    tip = np.array(s["robot"]["tool"])[:3, 3]
    assert np.linalg.norm(np.array(s["object"]["T"])[:3, 3] - tip) < 0.002
    h.ok("jog_joint", joint=0, delta=10)
    h.idle()
    s = h.twin.snapshot()
    assert np.linalg.norm(np.array(s["object"]["T"])[:3, 3] - np.array(s["robot"]["tool"])[:3, 3]) < 0.002
    h.refused("set_object", "holding", xy=[-0.4, 0.1])
    h.close()
    print("OK: placed at the spot; a miss goes home empty; a held object rides with the arm\n")


def test_twin_detect():
    print("=" * 60)
    print("Detect: home, ready over MQTT, the vision PC's pose shown")
    print("=" * 60)
    h = Harness(fake_vision=True)
    h.ok("arm")
    h.ok("jog_joint", joint=0, delta=15)
    h.idle()
    h.ok("detect")
    h.idle(30)
    s = h.twin.snapshot()
    expect = np.array([-0.42, 0.12, 0.03])   # demo_pose_on_table: on the twin's table
    assert s["object"]["source"] == "camera" and np.allclose(np.array(s["object"]["T"])[:3, 3], expect, atol=1e-3)
    assert np.allclose(h.joints(), HOME, atol=0.3), "detect goes home first"
    h.close()
    print(f"OK: object from the camera at {np.round(expect, 4).tolist()}\n")


class SlowGripper(FakeGripper):
    """A gripper that takes a while to close, like the Feetech (0.25..3 s)."""

    def close(self, speed=50):
        time.sleep(0.8)
        return super().close(speed)


def test_stop_latches():
    print("=" * 60)
    print("STOP latches: during planning, during a gripper close, on disarm")
    print("=" * 60)
    # right after Pick, while the task thread still plans
    h = Harness(place=[-0.25, 0.35])
    h.ok("arm")
    h.ok("pick")
    h.ok("stop")
    h.idle(30)
    assert h.motions_after("Stop") == [], h.motions_after("Stop")
    h.refused("move_joints", "not armed", joints=HOME.tolist())   # latched until Arm
    h.close()

    # while the gripper closes at the grasp: nothing after it
    h = Harness(place=[-0.25, 0.35], gripper=SlowGripper(object_position=None))
    h.ok("arm")
    h.ok("pick")
    wait_for(lambda: any("at grasp" in line for line in h.lines), 30, "the grasp")
    time.sleep(0.2)   # inside the 0.8 s close
    h.ok("stop")
    h.idle(30)
    after = h.motions_after("Stop")
    assert after == [], after
    assert not h.twin.armed and any("stopped" in line for line in h.lines)
    h.close()

    # switching Arm off while a move runs stops it too
    h = Harness()
    h.ok("arm")
    h.ok("speed", value=5)
    h.ok("move_joints", joints=(HOME + [60, 0, 0, 0, 0, 0]).tolist())
    wait_for(lambda: h.fake.mode == 7, 2, "the move to start")
    h.ok("arm", on=False)
    h.idle(10)
    assert abs(h.joints()[0] - (HOME[0] + 60)) > 5 and h.fake.mode == 5
    h.ok("arm")   # Arm again clears the latch
    h.ok("home")
    h.idle()
    h.close()
    print("OK: no motion after STOP or disarm, even between moves; Arm re-arms\n")


def test_auto_disarm():
    print("=" * 60)
    print("Disarm when the robot stops being the app's to move")
    print("=" * 60)
    h = Harness()
    other = DobotTCP("127.0.0.1", h.fake.port).connect()   # the pendant / another program
    cases = [("disabled", lambda: other.disable(), lambda: other.enable()),
             ("drag mode", lambda: other.start_drag(), lambda: other.stop_drag()),
             ("someone else's MovJ", lambda: other.mov_j(HOME + [30, 0, 0, 0, 0, 0], v=20), lambda: None),
             ("pendant jog", lambda: h.fake.external_move(HOME, 0.6, wait=False), lambda: None)]
    for what, do, undo in cases:
        h.ok("arm")
        do()
        wait_for(lambda: not h.twin.armed, 2, f"disarm on {what}")
        wait_for(lambda: h.fake.mode in (4, 5, 6), 5, "still")
        undo()
        wait_for(lambda: h.fake.mode == 5, 5, "idle again")
    other.close()
    # feedback gone (cable): disarm
    h.ok("arm")
    h.feedback.stop()
    wait_for(lambda: not h.twin.armed, 2, "disarm without feedback")
    h.robot.close()
    h.fake.stop()
    print("OK: disabled, drag mode, someone else's motion, pendant jog, lost feedback -> disarmed\n")


def test_dropped_move():
    print("=" * 60)
    print("A move Stop() drops before it starts does not hang the wait")
    print("=" * 60)
    fake = FakeDobot(enabled=True, feedback=False).start()
    a = DobotTCP("127.0.0.1", fake.port, poll_s=0.01).connect()
    b = DobotTCP("127.0.0.1", fake.port).connect()   # someone else's Stop()
    a.mov_j(HOME + [90, 0, 0, 0, 0, 0], v=5)          # long first move
    second = a.mov_j(HOME, v=100)                     # queued behind it
    time.sleep(0.2)
    b.stop()
    t0 = time.monotonic()
    try:
        a.wait_motion(second, lambda q: True)
        raise AssertionError("waited for a move that was dropped")
    except DobotError as e:
        assert time.monotonic() - t0 < 5 and ("never started" in str(e) or "not at the target" in str(e)), e
    # our own Stop() ends our own wait at once
    cmd = a.mov_j(HOME + [90, 0, 0, 0, 0, 0], v=5)
    threading.Timer(0.3, a.stop).start()
    t0 = time.monotonic()
    try:
        a.wait_motion(cmd, lambda q: False)
        raise AssertionError("not stopped")
    except DobotError as e:
        assert str(e).endswith(": stopped") and time.monotonic() - t0 < 0.8, (e, time.monotonic() - t0)
    a.close()
    b.close()
    fake.stop()
    print("OK: a dropped move is reported in seconds, our own Stop() at once\n")


def test_bad_input():
    print("=" * 60)
    print("Bad input from a browser: refused, nothing breaks")
    print("=" * 60)
    h = Harness()
    for cmd, args in [("arm", {"on": "false"}), ("set_place", {"xy": [float("nan"), 0.3]}),
                      ("set_place", {"xy": [1]}), ("set_object", {"xy": {}}), ("set_object", {"xy": [0.1, 0.2], "diameter": 9}),
                      ("speed", {"value": float("inf")}), ("jog_joint", {"joint": float("inf"), "delta": 5}),
                      ("jog_joint", {"joint": True, "delta": 5}), ("move_joints", {"joints": [0] * 1000}),
                      ("preview_pose", {"T": [[1, 0, 0]]}), ("detect", {}), ("no_such", {}), ("arm", {"bogus": 1})]:
        r = h.twin.handle({"cmd": cmd, **args})
        assert not r["ok"], (cmd, args, r)
    assert not h.twin.armed and h.twin.place is None
    json.dumps(h.twin.snapshot(), allow_nan=False)   # the state still serialises for every browser
    h.close()
    print("OK: NaN, wrong types and shapes, unknown commands all refused with a reply\n")


def test_detect_safeguards():
    print("=" * 60)
    print("Detect: STOP ends the wait; a camera that moved is not trusted")
    print("=" * 60)
    h = Harness()                 # no vision link
    h.ok("arm")
    h.refused("detect", "no vision link")
    h.close()
    h = Harness()
    h.twin.can_detect = True      # a link nobody answers on
    h.ok("arm")
    h.ok("detect", timeout=60)
    wait_for(lambda: (h.twin.step or "").startswith("waiting"), 10, "the wait")
    t0 = time.monotonic()
    h.ok("stop")
    h.idle(10)
    assert time.monotonic() - t0 < 2
    h.close()

    h = Harness()
    h.twin.can_detect = True
    q0 = h.fake.joints.copy()
    ready = h.twin.link.subscribe(READY_TOPIC)

    def answer_late():   # the camera moved while it looked (no mode change: only the check sees it)
        ready.get(timeout=10)
        with h.fake.lock:
            h.fake.joints = q0 + [8, 0, 0, 0, 0, 0]
        time.sleep(0.05)
        h.twin.link.publish(POSE_TOPIC, {"request_id": None, "T_camera_object": np.eye(4).tolist()})

    threading.Thread(target=answer_late, daemon=True).start()
    h.ok("arm")
    h.ok("detect", timeout=20)
    h.idle(30)
    assert any("moved while the camera was looking" in line for line in h.lines), h.lines[-5:]
    assert h.twin.snapshot()["object"]["source"] == "manual"
    h.close()
    print("OK: STOP ends a Detect wait at once; a pose from a moved camera is refused\n")


def test_gripper_states():
    print("=" * 60)
    print("Gripper: opened before a pick; --no-gripper picks; a second close keeps the grip")
    print("=" * 60)
    h = Harness(place=[-0.25, 0.35])
    h.ok("arm")
    h.ok("gripper", action="close")      # on air
    h.idle()
    assert h.twin.grip["position"] == 255
    motions = lambda: sum(n in ("MovJ", "MovL") for n, _, _ in h.fake.log)   # noqa: E731
    before, opens, real_open = motions(), [], h.gripper.open
    h.gripper.open = lambda *a, **k: (opens.append(motions()), real_open(*a, **k))
    h.ok("pick")
    h.idle(90)
    assert opens and opens[0] == before, (opens, before)   # opened before the first motion of the pick
    assert any("[placed] at [-0.25, 0.35]" in line for line in h.lines)
    h.close()

    h = Harness(sim=False)                # real arm, simulated gripper
    h.ok("arm")
    h.ok("pick")
    h.idle(90)
    s = h.twin.snapshot()
    assert s["gripper"]["holding"] and s["object"]["attached"], s["gripper"]
    h.ok("gripper", action="close")       # again, while holding
    h.idle()
    assert h.twin.snapshot()["object"]["attached"] and h.twin.grip["holding"]
    h.refused("pick", "holding")
    h.close()
    print("OK: fingers opened before coming down; --no-gripper grips; holding survives a second close\n")


def test_line_with_reading_noise():
    print("=" * 60)
    print("Straight-line moves are not refused for joint reading noise")
    print("=" * 60)
    h = Harness()
    h.ok("arm")
    T = fk(joint_map.servo_to_visual(h.robot.get_angle()), "tool")
    from manipulator_kinematics import rot_x
    T[:3, :3] = T[:3, :3] @ rot_x(math.radians(0.02))   # 0.02 deg off: noise, not a turn
    T[2, 3] -= 0.02
    h.ok("preview_pose", T=T.tolist(), linear=True)
    T[:3, :3] = T[:3, :3] @ rot_x(math.radians(5))      # a real turn is still refused for a line
    h.refused("preview_pose", "orientation", T=T.tolist(), linear=True)
    h.close()
    print("OK\n")


def test_getpose_without_arguments():
    print("=" * 60)
    print("Firmware that refuses GetPose(user=0,tool=0) (ErrorID -40001)")
    print("=" * 60)
    from manipulator_kinematics import T_FLANGE_TOOL, pose
    for tool, ok in ((None, True), (T_FLANGE_TOOL, True), (pose(np.eye(3), [0, 0, 0.05]), False)):
        h = Harness(pose_args=False, pendant_tool=tool)
        r = h.twin.handle({"cmd": "arm"})
        assert r["ok"] == ok, (tool, r, h.lines[-2:])
        if ok:
            h.ok("pick")
            h.idle(60)
            assert any("at grasp" in line for line in h.lines) and h.twin.grip["holding"], h.lines[-4:]
        h.close()
    print("OK: arms and picks with the pendant on tool 0 or the 197 mm tool; refuses an unknown tool\n")


def test_v3_protocol():
    print("=" * 60)
    print("V3 firmware: motion on port 30003, no Stop() / MovJ(joint=...)")
    print("=" * 60)
    h = Harness(protocol="v3", pose_args=False, command_id=False)
    assert h.robot.protocol == "v3"
    h.ok("arm")
    start = h.joints()
    h.ok("jog_joint", joint=0, delta=10)
    h.idle(30)
    assert abs(h.joints()[0] - start[0] - 10) < 0.05, h.joints()
    h.ok("jog_tool", axis="z", delta=30)
    h.idle(60)
    names = [n for n, _, _ in h.fake.log]
    assert "JointMovJ" in names and "SpeedJ" in names, names[-10:]
    assert not any(n in ("MovJ", "MovL", "Stop") for n in names), [n for n in names if n in ("MovJ", "MovL", "Stop")]
    h.ok("speed", value=5)
    h.ok("move_joints", joints=(HOME + [90, 0, 0, 0, 0, 0]).tolist())
    wait_for(lambda: h.fake.mode == 7, 5, "the move to start")
    time.sleep(0.5)
    h.ok("stop")
    h.idle(10)
    assert "ResetRobot" in [n for n, _, _ in h.fake.log] and h.fake.mode == 5 and not h.twin.armed
    h.ok("arm")
    h.ok("speed", value=100)
    h.ok("home")
    h.idle(60)
    h.ok("pick")
    h.idle(90)
    assert any("at grasp" in line for line in h.lines), h.lines[-4:]
    h.close()
    print("OK: jog, straight line, STOP (ResetRobot) and pick over JointMovJ\n")


def camera_position(h, xyz_base, w=0.0):
    """Where the wrist camera sees a base-frame point from where the arm is now: [x, y, z, w]."""
    T_cam = fk(joint_map.servo_to_visual(h.robot.get_angle()), "camera")
    p = np.linalg.inv(T_cam) @ np.array([*xyz_base, 1.0])
    return [*p[:3].round(6).tolist(), w]


def test_object_from_camera():
    print("=" * 60)
    print("Object at x, y, z, w in the camera frame (the vision agent's numbers)")
    print("=" * 60)
    h = Harness(place=[-0.25, 0.35])
    r = h.ok("set_object_camera", position=camera_position(h, [-0.40, 0.10, 0.03]), product="tomato")
    assert np.allclose(r["xyz"], [-0.40, 0.10, 0.03], atol=1e-4) and abs(r["above_table_mm"] - 30) < 0.1, r
    obj = h.twin.snapshot()["object"]
    assert obj["source"] == "camera: tomato" and np.allclose(np.array(obj["T"])[:3, 3], [-0.40, 0.10, 0.03], atol=1e-4)
    for bad, why in (([0.1, 0.0, 300.0, 0], "meters"), ([0.0, 0.0, -0.2, 0], "in front of the camera"),
                     ([0.0, 0.0], "position must have"), ([0, 0, "x", 0], "numbers"), ([0, 0, 0.3, 720], "degrees")):
        h.refused("set_object_camera", why, position=bad)
    h.refused("set_object_camera", "not a unit quaternion", position=[0, 0, 0.3], orientation=[0, 0, 0, 0.1])

    # FoundationPose's pose: position + quaternion (xyzw, camera frame) -> the object takes both
    from manipulator_kinematics import pose, rot_x, rot_z
    from mqtt_link import matrix_quaternion
    T_base = pose(rot_z(math.radians(40)) @ rot_x(math.radians(15)), [-0.41, 0.11, 0.03])
    T_cam = np.linalg.inv(fk(joint_map.servo_to_visual(h.robot.get_angle()), "camera")) @ T_base
    quat = matrix_quaternion(T_cam[:3, :3]).tolist()
    for args in ({"position": T_cam[:3, 3].tolist(), "orientation": quat},
                 {"position": T_cam[:3, 3].tolist() + quat}):                 # the 7-number form
        r = h.ok("set_object_camera", **args)
        T = np.array(h.twin.snapshot()["object"]["T"])
        assert np.allclose(T, T_base, atol=1e-5), (T, T_base)
        assert abs(r["yaw_deg"] - 40) < 0.1 and abs(r["tilt_deg"] - 15) < 0.1, r
    assert h.twin.object["symmetric"] is False   # a real orientation: the grasp follows it

    h.ok("set_object_camera", position=T_cam[:3, 3].tolist(), orientation=quat, symmetric=True)
    h.ok("arm")
    h.ok("pick")
    h.idle(90)
    assert any("[placed]" in line for line in h.lines)

    # a turned (not round) object: the fingers close across its x axis, as it is turned
    T_base = pose(rot_z(math.radians(30)), [-0.42, 0.12, 0.03])
    T_cam = np.linalg.inv(fk(joint_map.servo_to_visual(h.robot.get_angle()), "camera")) @ T_base
    h.ok("set_object_camera", position=T_cam[:3, 3].tolist(), orientation=matrix_quaternion(T_cam[:3, :3]).tolist())
    grasp = next(m for m in h.ok("plan")["plan"]["moves"] if m["name"] == "grasp")
    T_grasp = fk(joint_map.servo_to_visual(grasp["q"]), "tool")
    yaw = math.degrees(math.atan2(T_grasp[1, 0], T_grasp[0, 0]))
    assert min(abs((yaw - a + 180) % 360 - 180) for a in (30, 210)) < 1.0, yaw
    h.close()
    print("OK: position + quaternion put the object where and as FoundationPose sees it; "
          "the grasp follows a turned object; mm, behind the camera, bad w / quaternion refused\n")


def test_mqtt_commands():
    print("=" * 60)
    print("MQTT commands from the vision agent: idle / start pick / start place / stop")
    print("=" * 60)
    from mqtt_link import COMMAND_TOPIC, STATUS_TOPIC, command_message, parse_command
    h = Harness(fake_vision=True)
    replies = h.twin.link.subscribe(STATUS_TOPIC)

    def send(msg):
        while not replies.empty():
            replies.get_nowait()
        h.twin.link.publish(COMMAND_TOPIC, msg)

    def reply(*states, timeout=90):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                r = json.loads(replies.get(timeout=0.2))
            except Exception:
                continue
            if "action" in r and r["state"] in states:   # the twin's replies, not the pick's own status
                return r
        raise AssertionError(f"no {states} reply")

    # the message as written in the request, the "idal" spelling too
    c = parse_command('{"state": "idal", "product": "", "action": "pick", "position": "0.02, -0.03, 0.30, 1"}')
    assert c["state"] == "idle" and np.allclose(c["position"], [0.02, -0.03, 0.30, 1])
    assert parse_command('{"state":"start","action":"pick","position":{"x":0,"y":0,"z":0.3}}')["position"] == [0, 0, 0.3]

    # FoundationPose's orientation over MQTT: the twin's object turns with it
    from manipulator_kinematics import pose, rot_z
    from mqtt_link import matrix_quaternion
    T_base = pose(rot_z(math.radians(-50)), [-0.40, 0.10, 0.03])
    T_cam = np.linalg.inv(fk(joint_map.servo_to_visual(h.robot.get_angle()), "camera")) @ T_base
    send(command_message("idle", None, T_cam[:3, 3], "tomato", matrix_quaternion(T_cam[:3, :3])))
    r = reply("updated", "refused", timeout=10)
    assert r["state"] == "updated", r
    assert np.allclose(np.array(h.twin.snapshot()["object"]["T"]), T_base, atol=1e-5)

    target = [-0.42, 0.12, 0.03]
    send(command_message("idle", "pick", camera_position(h, target), "tomato"))
    r = reply("updated", "refused", timeout=10)
    assert r["state"] == "updated" and r["product"] == "tomato", r
    assert np.allclose(np.array(h.twin.snapshot()["object"]["T"])[:3, 3], target, atol=1e-4)

    send(command_message("start", "pick", camera_position(h, target), "tomato"))
    r = reply("refused", "accepted", timeout=10)
    assert r["state"] == "refused" and "not armed" in r["detail"], r

    send('{"state": "start", "action": "jump"}')
    assert "pick or place" in reply("refused", timeout=10)["detail"]
    send({"state": "start", "action": "place"})
    assert "needs a position" in reply("refused", timeout=10)["detail"]

    h.ok("arm")
    send({**command_message("start", "pick", camera_position(h, target), "tomato"), "request_id": "r1"})
    r = reply("accepted", "refused", timeout=10)
    assert r["state"] == "accepted", r
    r = reply("done", "failed", "stopped")
    assert r["state"] == "done" and r["request_id"] == "r1", r
    h.idle()
    s = h.twin.snapshot()
    assert s["gripper"]["holding"] and s["object"]["attached"] and np.allclose(h.joints(), HOME, atol=0.3)

    place = [-0.25, 0.35]
    send(command_message("start", "place", camera_position(h, [*place, 0.03]), "tomato"))
    r = reply("accepted", "refused", timeout=10)
    assert r["state"] == "accepted", r
    r = reply("done", "failed", "stopped")
    assert r["state"] == "done", r
    h.idle()
    s = h.twin.snapshot()
    T = np.array(s["object"]["T"])
    assert not s["gripper"]["holding"] and np.allclose(T[:3, 3], [*place, 0.03], atol=0.002), T[:3, 3]

    # stop: during a pick, the arm stops, the twin disarms, the sender hears "stopped"
    h.ok("speed", value=5)
    send(command_message("start", "pick", camera_position(h, [*place, 0.03]), "tomato"))
    r = reply("accepted", "refused", timeout=10)
    assert r["state"] == "accepted", r
    wait_for(lambda: h.fake.mode == 7, 5, "the pick to start moving")
    send("stop")
    assert reply("stopped", timeout=10)
    h.idle(10)
    assert not h.twin.armed and h.fake.mode == 5
    h.close()
    print("OK: idle shows the object, start pick / place run and report done, stop stops; bad ones refused\n")


def test_table_height_checks():
    print("=" * 60)
    print("Table height: lowest point, arming refuses an arm already below it, CLI guard")
    print("=" * 60)
    import subprocess
    import sys
    from manipulator_kinematics import lowest_point
    z_home, part = lowest_point(joint_map.servo_to_visual(HOME))
    h = Harness(table_z=z_home + 0.01)   # a table height above where the arm already is
    assert abs(h.twin.snapshot()["robot"]["lowest"][0] - z_home) < 1e-3
    h.refused("arm", "already below the table height")
    h.close()
    h = Harness(table_z=z_home - 0.06)
    h.ok("arm")
    h.close()
    r = subprocess.run([sys.executable, "digital_twin.py", "--robot", "127.0.0.1", "--table-z", "-0.10"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "--pedestal" in r.stderr, r.stderr
    r = subprocess.run([sys.executable, "pick_runner.py", "--robot-ip", "127.0.0.1", "--table-z", "-0.10"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "--pedestal" in r.stderr, r.stderr
    print(f"OK: lowest point at home {z_home * 1000:.0f} mm ({part}); -0.10 without --pedestal refused\n")


# ============================================================
# WEBSOCKET AND BROWSER
# ============================================================

def test_camera_calibration():
    print("=" * 60)
    print("Camera calibration: a camera turned 88 deg on its mount, found from camera / tool tip pairs")
    print("=" * 60)
    import manipulator_kinematics as mk
    from camera_calibration import CalibrationError, solve
    from manipulator_kinematics import pose, rot_x, rot_z
    # the "real" camera: the table's, turned 88 deg about the view axis, tilted 4 deg, shifted
    C = pose(rot_z(math.radians(88)) @ rot_x(math.radians(4)), [-0.012, -0.015, 0.002])
    T_true = mk.T_FLANGE_CAMERA_DH @ C
    objects = [[-0.39, -0.06, 0.04], [-0.51, 0.07, 0.095], [-0.49, 0.26, 0.09], [-0.40, 0.12, 0.09], [-0.30, 0.20, 0.05]]
    home = joint_map.home_servo_deg
    T_bf = fk(joint_map.servo_to_visual(home), "flange")

    def seen(obj, T_base_flange=T_bf):   # what FoundationPose reports for an object at obj (base frame)
        return (np.linalg.inv(T_base_flange @ T_true) @ np.array([*obj, 1.0]))[:3].round(5).tolist()

    # by hand: joints known, or the twin's object_base (placed with the camera table)
    by_joints = [{"camera": seen(o), "joints": home.tolist(), "tip_mm": (np.array(o) * 1000).tolist()} for o in objects]
    r = solve(by_joints)
    assert np.allclose(r["T_flange_camera"], T_true, atol=2e-4) and max(r["residual_mm"]) < 0.05, r
    assert abs(r["turn_deg"] - 88) < 0.5 and max(r["camera_table_error_mm"]) > 30, r
    T_bc_table = T_bf @ mk.T_FLANGE_CAMERA_DH
    by_object = [{"camera": p["camera"], "tip_mm": p["tip_mm"],
                  "object_base_mm": ((T_bc_table @ np.array([*p["camera"], 1.0]))[:3] * 1000).tolist()} for p in by_joints]
    by_object[0]["object_base_mm"][0] += 30   # one typing slip: found, and not used for the camera pose
    r = solve(by_object)
    assert np.allclose(r["T_flange_camera"], T_true, atol=1e-4) and any("pair 1" in n for n in r["notes"]), r
    for bad, why in ((by_joints[:2], "at least 3"),
                     ([dict(p, camera=[0.0, 0.0, 0.3 + 0.01 * i]) for i, p in enumerate(by_joints)], "in a line"),
                     ([dict(by_joints[0], joints=None)] + by_joints[1:], "joints or object_base")):
        try:
            solve(bad)
            raise AssertionError(f"accepted: {why}")
        except CalibrationError as e:
            assert why in str(e), e

    # in the twin: type the numbers, record; put the tool tip on it, record; ... solve, apply
    h = Harness()
    saved_file = mk.CAMERA_CALIBRATION_FILE
    mk.CAMERA_CALIBRATION_FILE = os.path.join(tempfile.mkdtemp(), "camera_calibration.json")
    try:
        h.ok("arm")
        h.ok("speed", value=100)
        h.refused("calib_tip", "camera point first")
        for obj in objects:
            h.ok("home")
            h.idle(60)
            h.ok("set_object_camera", position=seen(obj))
            h.ok("calib_point")
            h.refused("calib_tip", "has not moved")
            T = fk(joint_map.servo_to_visual(home), "tool")
            T[:3, 3] = obj
            h.ok("move_pose", T=T.tolist())
            h.idle(60)
            h.ok("calib_tip")
        assert h.twin.snapshot()["calib"] == {"pairs": 5, "pending": False, "active": False}
        r = h.ok("calib_solve")["result"]
        assert np.allclose(r["T_flange_camera"], T_true, atol=1e-4) and max(r["residual_mm"]) < 0.5, r
        assert h.ok("calib_apply")["reload"] and os.path.exists(mk.CAMERA_CALIBRATION_FILE)
        assert h.twin.snapshot()["calib"]["active"] and np.allclose(mk.T_FLANGE_CAMERA, T_true, atol=1e-4)
        # now the camera's numbers put the tomato where it really is
        h.ok("home")
        h.idle(60)
        xyz = h.ok("set_object_camera", position=seen(objects[2]))["xyz"]
        assert np.allclose(xyz, objects[2], atol=1e-3), xyz
        h.ok("calib_remove")
        assert mk.camera_calibration is None and np.allclose(mk.T_FLANGE_CAMERA, mk.T_FLANGE_CAMERA_DH)
        assert not os.path.exists(mk.CAMERA_CALIBRATION_FILE)
    finally:
        mk.set_camera_calibration(None)
        mk.CAMERA_CALIBRATION_FILE = saved_file
        h.close()
    print("OK: an 88 deg camera turn found from 5 pairs (by joints, by the twin's numbers, recorded in the "
          "twin); after Apply the camera's numbers land on the real object\n")


def test_live_camera():
    print("=" * 60)
    print("Live camera: /camera/info and the MJPEG stream (simulated camera)")
    print("=" * 60)
    from aiohttp.test_utils import TestClient, TestServer
    from camera_stream import FakeCamera
    try:
        import cv2  # noqa: F401  (the camera's JPEGs)
    except ImportError:
        print("skipped: needs opencv-python\n")
        return
    h = Harness()
    cam = FakeCamera(fps=30).start()

    async def run():
        # without --camera: the page is told there is none
        async with TestClient(TestServer(make_app(h.twin, load_model(), CONFIG, TOKEN))) as client:
            info = await (await client.get(f"/camera/info?token={TOKEN}")).json()
            assert info["available"] is False and "--camera" in info["error"], info
            assert (await client.get(f"/camera.mjpg?token={TOKEN}")).status == 404
        async with TestClient(TestServer(make_app(h.twin, load_model(), CONFIG, TOKEN, camera=cam,
                                                  camera_fps=30))) as client:
            for path in ("/camera/info", "/camera.mjpg", f"/camera.mjpg?token=wrong-token"):
                assert (await client.get(path)).status == 403, path
            r = await client.get(f"/camera.mjpg?token={TOKEN}", headers={"Origin": "http://evil.example"})
            assert r.status == 403
            assert (await client.get(f"/camera.mjpg?token={TOKEN}&kind=infrared")).status == 400
            wait_for(lambda: cam.info()["live"], 5, "the camera's first frame")
            info = await (await client.get(f"/camera/info?token={TOKEN}")).json()
            assert info["available"] and info["live"] and info["intrinsics"]["fx"] > 0, info
            for kind in ("color", "depth"):
                r = await client.get(f"/camera.mjpg?token={TOKEN}&kind={kind}")
                assert r.status == 200 and r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
                data, t0 = b"", time.monotonic()
                while data.count(b"\xff\xd9") < 3 and time.monotonic() - t0 < 5:   # three whole JPEGs
                    data += await r.content.readany()
                r.close()
                assert data.startswith(b"--frame\r\nContent-Type: image/jpeg") and data.count(b"\xff\xd8") >= 3, data[:80]
                import cv2
                start = data.index(b"\xff\xd8")
                img = cv2.imdecode(np.frombuffer(data[start:data.index(b"\xff\xd9", start) + 2], np.uint8), 1)
                assert img.shape == (480, 640, 3), img.shape

    try:
        asyncio.run(run())
    finally:
        cam.stop()
        h.close()
    print("OK: the stream needs the token, plays colour and depth JPEGs; none without --camera\n")


def test_websocket():
    print("=" * 60)
    print("WebSocket: hello, 30 Hz state, commands, deadman stop")
    print("=" * 60)
    from aiohttp.test_utils import TestClient, TestServer
    h = Harness()
    config = {"sim": True, "robot": None, "table_z": 0.0, "detect": False, "max_speed": 100, "gripper": "simulated"}

    async def run():
        async with TestClient(TestServer(make_app(h.twin, load_model(), config, TOKEN))) as client:
            for path in ("/", "/web/app.js", "/web/vendor/three/three.module.min.js", "/web/model/nova2/Link1.STL"):
                r = await client.get(path)
                assert r.status == 200, (path, r.status)
            # no token, or a page from another site: denied
            for kw in ({"path": "/ws"}, {"path": f"/ws?token={TOKEN}", "headers": {"Origin": "http://evil.example"}}):
                bad = await client.ws_connect(kw["path"], headers=kw.get("headers"))
                assert (await bad.receive_json())["type"] == "denied"
                await bad.receive()
                assert bad.close_code == 4003, bad.close_code
            ws = await client.ws_connect(f"/ws?token={TOKEN}")
            hello = await ws.receive_json()
            assert hello["type"] == "hello" and len(hello["model"]["joints"]) == 6
            t0, states = time.monotonic(), 0
            while time.monotonic() - t0 < 1.0:
                msg = await ws.receive_json()
                states += msg["type"] == "state"
            assert 20 <= states <= 40, states

            async def call(cmd, **args):
                await ws.send_json({"cmd": cmd, "id": cmd, **args})
                while True:
                    msg = await ws.receive_json()
                    if msg["type"] == "reply" and msg["id"] == cmd:
                        return msg

            assert (await call("arm", on=True))["ok"]
            target = (HOME + [10, 0, 0, 0, 0, 0]).tolist()
            assert (await call("move_joints", joints=target))["ok"]
            while True:
                msg = await ws.receive_json()
                if msg["type"] == "state" and not msg["task"] and np.allclose(msg["robot"]["joints"], target, atol=0.3):
                    break
            await ws.send_str("not json")
            while (msg := await ws.receive_json())["type"] != "reply":
                pass
            assert not msg["ok"]
            assert not (await call("no_such_command"))["ok"]
            # deadman: a move is running and the last browser goes away -> STOP
            await call("speed", value=5)
            assert (await call("move_joints", joints=(HOME + [80, 0, 0, 0, 0, 0]).tolist()))["ok"]
            await asyncio.sleep(0.3)
            await ws.close()
            for _ in range(100):
                await asyncio.sleep(0.05)
                if h.twin.task is None:
                    break
        return states

    states = asyncio.run(run())
    q = h.joints()
    assert not h.twin.armed and abs(q[0] - (HOME[0] + 80)) > 5, q
    h.close()
    print(f"OK: {states} states/s; commands and errors replied; closing the page stopped the arm\n")


def serve_in_thread(app):
    from aiohttp import web
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(app)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(10)
    return port, lambda: loop.call_soon_threadsafe(loop.stop)


def chromium():
    for path in ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome", os.environ.get("CHROMIUM", "")):
        if path and os.path.exists(path):
            return path
    return None


def test_browser():
    print("=" * 60)
    print("Real browser: the page mirrors the robot and moves it")
    print("=" * 60)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("SKIPPED: pip install playwright to run the browser test\n")
        return
    h = Harness(place=[-0.25, 0.35])
    config = {"sim": True, "robot": None, "table_z": 0.0, "detect": False, "max_speed": 100, "gripper": "simulated"}
    port, stop = serve_in_thread(make_app(h.twin, load_model(), config, TOKEN))
    shots = tempfile.mkdtemp(prefix="twin_")
    with sync_playwright() as p:
        kw = {"executable_path": chromium()} if chromium() else {}
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"], **kw)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{port}/?token={TOKEN}")
        page.wait_for_function("window.twin && window.twin.meshesLoaded >= 7 && window.twin.state", timeout=30000)

        def scene_joints():
            return np.array(page.evaluate("window.twin.robotJoints()"))

        # robot -> page
        q = HOME + [30, -15, 20, 0, 10, -40]
        h.fake.external_move(q, 0.3)
        page.wait_for_timeout(300)
        assert np.allclose(scene_joints(), q, atol=1e-3), scene_joints()
        h.fake.external_move(HOME, 0.3)
        page.wait_for_timeout(300)
        # page -> robot: arm, jog J1 with its button
        page.check("#arm")
        page.wait_for_function("window.twin.state.armed")
        page.locator(".joint").nth(0).locator("button").nth(1).click()
        wait_for(lambda: abs(h.fake.joints[0] - (HOME[0] + 5)) < 0.01 and h.twin.task is None, 10, "the jog")
        page.wait_for_timeout(200)
        assert abs(scene_joints()[0] - (HOME[0] + 5)) < 0.01
        # put the object by clicking on the table: Top view, click near the middle
        page.click("[data-view='top']")
        page.click("#set-object")
        box = page.locator("#view canvas").bounding_box()
        page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        page.wait_for_function("window.twin.state.object && window.twin.state.object.source === 'manual'")
        page.screenshot(path=os.path.join(shots, "twin.png"))
        # a joint field takes typing while the robot's state streams in
        num = page.locator(".joint").nth(1).locator("input[type=number]")
        num.click(click_count=3)
        num.type("12.5", delay=80)
        num.press("Enter")
        page.wait_for_function("window.twin.state.ghost && Math.abs(window.twin.state.ghost[1] - 12.5) < 1e-6")
        page.click("#joints-reset")
        # a pick from the button that lasts well over the 3 s deadman: the page's keepalive covers it
        page.evaluate("window.twin.send('set_object', {xy: [-0.42, 0.12]})")
        page.evaluate("window.twin.send('speed', {value: 20})")
        t0 = time.monotonic()
        page.click("#pick")
        page.wait_for_function("window.twin.state.task", timeout=5000)
        page.wait_for_function("!window.twin.state.task", timeout=90000)
        took = time.monotonic() - t0
        assert took > 4 and any("[placed]" in line for line in h.lines), (took, h.lines[-6:])
        assert not any("no browser heard" in line for line in h.lines)
        assert not errors, errors
        browser.close()
    stop()
    h.close()
    print(f"OK: page follows the robot, its buttons move the robot, clicks place the object ({shots}/twin.png)\n")


if __name__ == "__main__":
    test_feedback_packets()
    test_feedback_follows_robot()
    test_model()
    test_twin_refuses_unsafe()
    test_twin_two_way()
    test_twin_stop()
    test_twin_pick_and_place()
    test_twin_detect()
    test_stop_latches()
    test_auto_disarm()
    test_dropped_move()
    test_bad_input()
    test_detect_safeguards()
    test_gripper_states()
    test_line_with_reading_noise()
    test_getpose_without_arguments()
    test_v3_protocol()
    test_table_height_checks()
    test_object_from_camera()
    test_mqtt_commands()
    test_camera_calibration()
    test_live_camera()
    test_websocket()
    test_browser()
