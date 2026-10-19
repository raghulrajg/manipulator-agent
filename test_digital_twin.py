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
import os
import socket
import struct
import tempfile
import threading
import time

import numpy as np

from digital_twin import WEB, Twin, load_model, make_app, model_fk, T_LINK6_FLANGE
from dobot_feedback import FIELDS, PACKET_SIZE, FeedbackError, FeedbackReader, pack_feedback, parse_feedback
from dobot_tcp import DobotTCP
from fake_dobot import FakeDobot
from gripper import FakeGripper
from manipulator_kinematics import SERVO_LIMITS_DEG, TableConstraint, fk, joint_map
from mqtt_link import LocalBus
from pick_runner import DEMO_T_CAMERA_OBJECT, start_fake_vision

HOME = joint_map.home_servo_deg


def wait_for(cond, timeout=10.0, what="condition"):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


class Harness:
    """A twin on a fake robot, in this process."""

    def __init__(self, table_z=0.0, place=None, fake_vision=False, **fake_kw):
        self.fake = FakeDobot(enabled=fake_kw.pop("enabled", True), **fake_kw).start()
        self.robot = DobotTCP("127.0.0.1", self.fake.port, poll_s=0.01).connect()
        self.feedback = FeedbackReader("127.0.0.1", self.fake.feedback_port).start()
        self.feedback.wait_first()
        link = LocalBus()
        if fake_vision:
            start_fake_vision(link, DEMO_T_CAMERA_OBJECT, delay=0.05)
        self.lines = []
        self.twin = Twin(self.robot, self.feedback, FakeGripper(object_position=None), link, table_z=table_z,
                         max_speed=100, place_xy=place, sim=True, log=self.lines.append)
        self.twin.fake = self.fake
        self.twin.check_model()
        self.twin.cmd_speed(100)

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
    low = below_table_target(h.twin.table_z)
    h.refused("move_joints", "below the table", joints=low.tolist())
    h.refused("preview_joints", "below the table", joints=low.tolist())
    h.refused("jog_tool", "at most 100 mm", axis="z", delta=-150)
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
    expect = (fk(np.zeros(6), "camera") @ DEMO_T_CAMERA_OBJECT)[:3, 3]
    assert s["object"]["source"] == "camera" and np.allclose(np.array(s["object"]["T"])[:3, 3], expect, atol=1e-3)
    assert np.allclose(h.joints(), HOME, atol=0.3), "detect goes home first"
    h.close()
    print(f"OK: object from the camera at {np.round(expect, 4).tolist()}\n")


# ============================================================
# WEBSOCKET AND BROWSER
# ============================================================

def test_websocket():
    print("=" * 60)
    print("WebSocket: hello, 30 Hz state, commands, deadman stop")
    print("=" * 60)
    from aiohttp.test_utils import TestClient, TestServer
    h = Harness()
    config = {"sim": True, "robot": None, "table_z": 0.0, "detect": False, "max_speed": 100, "gripper": "simulated"}

    async def run():
        async with TestClient(TestServer(make_app(h.twin, load_model(), config))) as client:
            for path in ("/", "/web/app.js", "/web/vendor/three/three.module.min.js", "/web/model/nova2/Link1.STL"):
                r = await client.get(path)
                assert r.status == 200, (path, r.status)
            ws = await client.ws_connect("/ws")
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
    port, stop = serve_in_thread(make_app(h.twin, load_model(), config))
    shots = tempfile.mkdtemp(prefix="twin_")
    with sync_playwright() as p:
        kw = {"executable_path": chromium()} if chromium() else {}
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"], **kw)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{port}/")
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
    test_websocket()
    test_browser()
