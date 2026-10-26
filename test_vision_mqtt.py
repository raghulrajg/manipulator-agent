"""
test_vision_mqtt.py
--------------------
Runs foundationpose_mqtt.py's real main loop with the camera, YOLO, SAM,
FoundationPose, OpenCV and the broker replaced by small fakes (no GPU,
no camera), and checks the MQTT behaviour:

  - nothing is detected or sent before a ready message
  - after a ready: detect -> register -> track, and the pose is sent once,
    only after 2.5 s of steady tracking, with the ready's request_id
  - a repeated ready gets the same pose again (the first one was lost) if
    it was sent less than 10 s ago, and a fresh detection otherwise; a new
    ready starts a fresh detection
  - losing the tomato, or holding its last pose while depth drops out,
    restarts the 2.5 s

Run: python3 test_vision_mqtt.py
"""

import importlib
import json
import math
import sys
import types

import numpy as np

from mqtt_link import POSE_TOPIC, READY_TOPIC, LocalBus, ready_message

FRAME_DT = 0.1                                 # fake camera: 10 fps
K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
TOMATO = np.array([0.03, -0.02, 0.38])         # tomato center in the camera frame [m]
DIAMETER = 0.06


class World:
    """What the fake camera sees, frame by frame."""

    def __init__(self, script):
        self.frame = 0
        self.script = script   # {frame: callable}
        self.hidden = set()    # frames the tomato is out of view
        self.holes = set()     # frames with no depth (too close): the script holds its last pose
        self.detect_frames = []
        self.bus = LocalBus()


WORLD = None


# ------------------------------------------------------------ fakes
class Rot:
    """Enough of scipy's Rotation for the vision code."""

    def __init__(self, R):
        self.R = np.asarray(R, float)

    @classmethod
    def from_matrix(cls, R):
        return cls(R)

    @classmethod
    def from_rotvec(cls, v):
        v = np.asarray(v, float)
        a = np.linalg.norm(v)
        if a < 1e-12:
            return cls(np.eye(3))
        k = v / a
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return cls(np.eye(3) + math.sin(a) * Kx + (1 - math.cos(a)) * Kx @ Kx)

    def inv(self):
        return Rot(self.R.T)

    def __mul__(self, other):
        return Rot(self.R @ other.R)

    def as_matrix(self):
        return self.R.copy()

    def as_rotvec(self):
        a = math.acos(max(-1.0, min(1.0, (np.trace(self.R) - 1) / 2)))
        if a < 1e-9:
            return np.zeros(3)
        w = np.array([self.R[2, 1] - self.R[1, 2], self.R[0, 2] - self.R[2, 0], self.R[1, 0] - self.R[0, 1]])
        return w / (2 * math.sin(a)) * a

    def as_quat(self):
        v = self.as_rotvec()
        a = np.linalg.norm(v)
        axis = v / a if a > 0 else np.zeros(3)
        return np.array([*(axis * math.sin(a / 2)), math.cos(a / 2)])

    def as_euler(self, seq, degrees=False):
        R = self.R
        e = np.array([math.atan2(R[2, 1], R[2, 2]), math.asin(-R[2, 0]), math.atan2(R[1, 0], R[0, 0])])
        return np.degrees(e) if degrees else e


class FakeCamera:
    def __init__(self, width=640, height=480, fps=30, host=None, port=None):
        self.K, self.depth_scale = K, 0.001

    def get_frames(self):
        WORLD.frame += 1
        action = WORLD.script.get(WORLD.frame)
        if action:
            action()
        color = np.zeros((480, 640, 3), np.uint8)
        depth = np.full((480, 640), 0.0 if WORLD.frame in WORLD.holes else TOMATO[2] - DIAMETER / 2)
        return color, depth

    def stop(self):
        pass


def tomato_bbox():
    u = K[0, 0] * TOMATO[0] / TOMATO[2] + K[0, 2]
    v = K[1, 1] * TOMATO[1] / TOMATO[2] + K[1, 2]
    r = K[0, 0] * DIAMETER / 2 / TOMATO[2]
    return [int(u - r), int(v - r), int(u + r), int(v + r)]


class Detector:
    def __init__(self, conf=0.25):
        pass

    def set_prompt(self, prompt):
        pass

    def detect(self, image):
        WORLD.detect_frames.append(WORLD.frame)
        return [] if WORLD.frame in WORLD.hidden else [{"bbox": tomato_bbox(), "conf": 0.9}]


class Segmenter:
    def __init__(self, *args, **kwargs):
        pass

    def segment(self, image, bbox):
        mask = np.zeros(image.shape[:2], bool)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = True
        return mask


class Pose6D:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeEstimator:
    diameter = DIAMETER

    def __init__(self):
        self.registrations = []

    def _pose(self):
        T = np.eye(4)
        T[:3, :3] = Rot.from_rotvec([0.3, -0.2, 0.5]).as_matrix()   # locked to identity by the script
        T[:3, 3] = TOMATO + np.random.default_rng(WORLD.frame).normal(0, 0.0005, 3)
        return T

    def register(self, K, rgb, depth, ob_mask, iteration):
        self.registrations.append(WORLD.frame)
        return self._pose()

    def track_one(self, rgb, depth, K, iteration):
        return self._pose()


class FakeLink(LocalBus):
    host, port = "fake", 0

    def __init__(self, client_id=None):
        super().__init__()
        WORLD.bus = self


def load_vision_module():
    def stub(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod

    cv2 = stub("cv2", FONT_HERSHEY_SIMPLEX=0, COLOR_BGR2RGB=4,
               putText=lambda *a, **k: None, rectangle=lambda *a, **k: None,
               imshow=lambda *a, **k: None, destroyAllWindows=lambda: None,
               cvtColor=lambda img, code: img, addWeighted=lambda a, wa, b, wb, g: a)
    cv2.waitKey = lambda ms: 27 if WORLD.frame >= WORLD.last_frame else -1
    stub("torch", cuda=types.SimpleNamespace(empty_cache=lambda: None, mem_get_info=lambda: (4e9, 8e9)),
         as_tensor=np.asarray)
    stub("trimesh")
    stub("scipy")
    stub("scipy.spatial")
    stub("scipy.spatial.transform", Rotation=Rot)
    stub("estimater", FoundationPose=None, ScorePredictor=None, PoseRefinePredictor=None)
    nv = stub("nvdiffrast")
    nv.torch = stub("nvdiffrast.torch")
    stub("remote_camera", CameraStreamer=FakeCamera)
    stub("precise_6dof_pose", Detector=Detector, Segmenter=Segmenter, Pose6D=Pose6D,
         draw_pose_axes=lambda display, *a, **k: display)
    sys.modules.pop("foundationpose_mqtt", None)
    fp = importlib.import_module("foundationpose_mqtt")
    fp.MqttLink = FakeLink
    fp.time = types.SimpleNamespace(monotonic=lambda: WORLD.frame * FRAME_DT)
    return fp


def run(script, last_frame, hidden=(), holes=()):
    global WORLD
    WORLD = World(script)
    WORLD.last_frame = last_frame
    WORLD.hidden, WORLD.holes = set(hidden), set(holes)
    fp = load_vision_module()
    est = FakeEstimator()
    fp.setup_foundation_pose = lambda path: (est, None)
    fp.print_6d_pose = lambda pose_data: None
    fp.main()
    return WORLD, est


def ready(request_id):
    return lambda: WORLD.bus.publish(READY_TOPIC, ready_message(request_id))


def test_ready_settle_publish():
    print("=" * 60)
    print("foundationpose_mqtt.py main loop with fake camera / models / broker")
    print("=" * 60)
    sent_at = []
    start = TOMATO.copy()
    moved = start + [0.03, 0.0, 0.0]

    def move_tomato():
        TOMATO[:] = moved

    # q1 repeated 2 s after its pose went out (lost on the way): same pose again;
    # q2 repeated 15 s after, the tomato moved in between: too old, detect again
    script = {10: ready("q1"), 60: ready("q1"), 70: ready("q2"), 200: move_tomato, 300: ready("q2")}

    # record the frame each pose goes out
    orig_publish = LocalBus.publish

    def publish(self, topic, payload):
        if topic == POSE_TOPIC:
            sent_at.append(WORLD.frame)
        orig_publish(self, topic, payload)

    LocalBus.publish = publish
    try:
        world, est = run(script, last_frame=360, hidden=range(85, 100), holes=range(115, 121))
    finally:
        LocalBus.publish = orig_publish
        TOMATO[:] = start

    poses = [json.loads(json.dumps(m)) for t, m in world.bus.sent if t == POSE_TOPIC]
    assert [p["request_id"] for p in poses] == ["q1", "q1", "q2", "q2"], poses
    assert min(world.detect_frames) >= 10, "detected before the ready"   # ready arrives in frame 10
    assert not [f for f in world.detect_frames if sent_at[0] < f < 70], "detected while idle after sending"
    assert not [f for f in world.detect_frames if sent_at[2] < f < 300], "detected while idle after sending"
    # the repeated q1 ready (the arm never got the pose) gets the same pose again, no new detection
    assert sent_at[1] == 60 and poses[1] == poses[0], (sent_at, poses[1])
    fresh = [(poses[0], sent_at[0], start), (poses[2], sent_at[2], start), (poses[3], sent_at[3], moved)]

    for p, f, where in fresh:
        reg = max(r for r in est.registrations if r < f)
        assert (f - reg) * FRAME_DT >= 2.5 - 1e-9, f"sent {(f - reg) * FRAME_DT:.1f} s after registering"
        T = np.array(p["T_camera_object"])
        assert np.allclose(T[:3, :3], np.eye(3)) and np.allclose(T[:3, 3], where, atol=0.003), (T, where)
        assert p["symmetric"] and p["origin"] == "center" and p["diameter"] == DIAMETER and p["object"] == "tomato"
    # the second request lost the tomato (frames 85-99) and was registered again after; then
    # depth dropped out (115-120, last pose held): the 2.5 s count only from live tracking after that
    assert any(100 <= r < 146 for r in est.registrations) and sent_at[2] >= 121 + 25, (est.registrations, sent_at)
    # the late q2 repeat was detected afresh, at the tomato's new place
    assert any(r >= 300 for r in est.registrations), est.registrations
    print(f"OK: ready at frames 10 / 60 (repeat, pose sent again) / 70 / 300 (repeat 15 s late, detected "
          f"again); registered at {est.registrations}; poses sent at frames {sent_at} (10 fps)\n")


def test_quaternions():
    """The quaternion <-> rotation conversions the twin uses for FoundationPose's
    orientation, including 180-degree turns (w = 0), which a camera looking
    straight down gives for an upright object."""
    import itertools
    import math
    from mqtt_link import matrix_quaternion, quaternion_matrix
    from manipulator_kinematics import rot_x, rot_z
    print("=" * 60)
    print("Quaternions (x, y, z, w) <-> rotations, 180-degree turns included")
    print("=" * 60)
    assert np.allclose(quaternion_matrix([0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]), rot_z(math.pi / 2))
    rotations = [rot_x(math.pi), rot_z(math.pi), rot_x(math.pi) @ rot_z(math.radians(30)),
                 rot_x(math.pi) @ rot_z(math.radians(-150))]
    # every axis permutation with signs that is a rotation (the 24 "square" turns)
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                R[row, col] = sign
            if np.linalg.det(R) > 0:
                rotations.append(R)
    rng = np.random.default_rng(3)
    for _ in range(2000):
        q = rng.normal(size=4)
        rotations.append(quaternion_matrix(q / np.linalg.norm(q)))
    for R in rotations:
        q = matrix_quaternion(R)
        assert np.isclose(np.linalg.norm(q), 1) and q[3] >= 0, q
        assert np.allclose(quaternion_matrix(q), R, atol=1e-9), (R, q)
    print(f"OK: {len(rotations)} rotations round-trip, 180-degree turns and the 24 square turns included\n")


if __name__ == "__main__":
    test_ready_settle_publish()
    test_quaternions()
