"""
mqtt_link.py
-------------
The MQTT side of the pick, shared by the vision PC (foundationpose_mqtt.py)
and the arm (pick_runner.py). Copy this file next to foundationpose_mqtt.py.

    arm    -> MQTT_READY_TOPIC  {"request_id", "joints", "stamp"}
              "arm is still at the detection pose, find the object"
              (a plain "ready" payload works too, e.g. from mosquitto_pub)
    vision -> MQTT_POSE_TOPIC   {"request_id", "T_camera_object" (4x4, m),
              "object", "origin", "symmetric", "diameter", "settle_s", "stamp"}
    arm    -> MQTT_STATUS_TOPIC {"request_id", "state", "detail", "stamp"}

Broker settings come from the same environment variables as the rest of
your MQTT stack: MQTT_BROKER_HOST (default localhost), MQTT_BROKER_PORT
(1883), MQTT_USERNAME, MQTT_PASSWORD; the three topics can be changed with
MQTT_READY_TOPIC, MQTT_POSE_TOPIC and MQTT_STATUS_TOPIC.

PoseGate is the vision side's "ready -> object found -> held still for
2.5 s -> send once" logic, kept here so it can be tested without a camera.
"""

import json
import os
import queue
import threading
import time

import numpy as np

READY_TOPIC = os.environ.get("MQTT_READY_TOPIC", "manipulator/ready")
POSE_TOPIC = os.environ.get("MQTT_POSE_TOPIC", "manipulator/object_pose")
STATUS_TOPIC = os.environ.get("MQTT_STATUS_TOPIC", "manipulator/status")

SETTLE_SECONDS = 2.5   # object tracked and still this long before its pose is sent
SETTLE_MOVE = 0.01     # [m] a pose that moves more than this restarts the wait


class MessageError(ValueError):
    pass


def _check_request_id(msg, text):
    rid = msg.setdefault("request_id", None)
    if not (rid is None or isinstance(rid, (str, int))) or isinstance(rid, bool):
        raise MessageError(f"request_id must be a string or a number: {text[:80]!r}")
    return msg


# ============================================================
# MESSAGES
# ============================================================

def ready_message(request_id, joints=None):
    msg = {"request_id": request_id, "stamp": time.time()}
    if joints is not None:
        msg["joints"] = np.round(np.asarray(joints, dtype=float), 4).tolist()
    return msg


def parse_ready(payload):
    """A ready message -> dict with request_id (None for a plain 'ready')."""
    text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)
    if text.strip().lower() in ("ready", "\"ready\""):
        return {"request_id": None}
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        raise MessageError(f"not a ready message: {text[:80]!r}") from None
    if not isinstance(msg, dict):
        raise MessageError(f"not a ready message: {text[:80]!r}")
    return _check_request_id(msg, text)


def pose_message(T_camera_object, request_id=None, object_name="object", origin="center",
                 diameter=None, settle_s=None, symmetric=False):
    """FoundationPose 4x4 (camera optical frame, meters) -> JSON-ready dict.
    origin says where the object frame sits: "center" (FoundationPose with
    a mesh centered on its bounding box) or "bottom". symmetric=True says
    the rotation carries no information (a round object, rotation locked),
    so the arm keeps its wrist turned as it is."""
    T = np.asarray(T_camera_object, dtype=float)
    msg = {"request_id": request_id, "object": object_name, "origin": origin,
           "symmetric": bool(symmetric), "frame": "camera",
           "T_camera_object": np.round(T, 6).tolist(), "stamp": time.time()}
    if diameter is not None:
        msg["diameter"] = round(float(diameter), 5)
    if settle_s is not None:
        msg["settle_s"] = round(float(settle_s), 3)
    return msg


def parse_pose_message(payload):
    """JSON from MQTT_POSE_TOPIC -> dict whose "T_camera_object" is a
    checked 4x4 numpy pose (rotation snapped to the nearest exact one).
    Raises MessageError if it is not a usable pose."""
    text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)
    try:
        msg = json.loads(text)
        T = np.asarray(msg["T_camera_object"], dtype=float)
    except (ValueError, KeyError, TypeError) as e:
        raise MessageError(f"bad pose message: {e}") from None
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise MessageError(f"T_camera_object must be a finite 4x4, got shape {T.shape}")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise MessageError(f"T_camera_object last row must be 0 0 0 1, got {T[3].tolist()}")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or np.linalg.det(R) < 0:
        raise MessageError("the 3x3 part of T_camera_object is not a rotation")
    U, _, Vt = np.linalg.svd(R)
    T[:3, :3] = U @ Vt
    msg["T_camera_object"] = T
    msg.setdefault("origin", "center")
    return _check_request_id(msg, text)


# ============================================================
# VISION SIDE: READY -> FOUND -> STILL FOR 2.5 S -> SEND
# ============================================================

class PoseGate:
    """
    arm(request_id) when a ready message arrives. Then call
    update(pose_or_None, now) every frame: it returns the pose to send once
    the object has been tracked for settle_s without moving more than
    max_move, and disarms. None (object lost) restarts the wait.
    """

    def __init__(self, settle_s=SETTLE_SECONDS, max_move=SETTLE_MOVE):
        self.settle_s, self.max_move = settle_s, max_move
        self.armed = False
        self.request_id = None
        self.answered = set()
        self._since = self._anchor = None

    def arm(self, request_id=None):
        """Start looking for request_id. Returns False (and changes nothing)
        for a repeat of the request already being worked on or answered --
        for an answered one, send the same pose again: the arm only repeats
        a ready while it is still waiting, so the first pose was lost."""
        if request_id is not None and (request_id in self.answered or
                                       (self.armed and request_id == self.request_id)):
            return False
        self.armed, self.request_id = True, request_id
        self._since = self._anchor = None
        return True

    def restart(self):
        self._since = self._anchor = None

    def remaining(self, now):
        """Seconds still to wait, None while no pose is being timed."""
        if not self.armed or self._since is None:
            return None
        return max(0.0, self.settle_s - (now - self._since))

    def update(self, pose_matrix, now):
        if not self.armed:
            return None
        if pose_matrix is None:
            self.restart()
            return None
        T = np.asarray(pose_matrix, dtype=float)
        if self._since is None or np.linalg.norm(T[:3, 3] - self._anchor) > self.max_move:
            self._since, self._anchor = now, T[:3, 3].copy()
            return None
        if now - self._since < self.settle_s:
            return None
        self.armed = False
        if self.request_id is not None:
            self.answered.add(self.request_id)
        return T.copy()


# ============================================================
# TRANSPORT
# ============================================================

class MqttLink:
    """paho-mqtt client: subscribe(topic) gives a queue.Queue of payloads
    (bytes), publish(topic, dict_or_text) sends JSON. Reconnects and
    re-subscribes on its own (paho's network thread)."""

    def __init__(self, client_id, host=None, port=None, username=None, password=None):
        import paho.mqtt.client as mqtt   # lazy, so the rest imports without paho
        self.host = host or os.environ.get("MQTT_BROKER_HOST", "localhost")
        self.port = int(port or os.environ.get("MQTT_BROKER_PORT", 1883))
        username = username if username is not None else os.environ.get("MQTT_USERNAME")
        password = password if password is not None else os.environ.get("MQTT_PASSWORD")
        if hasattr(mqtt, "CallbackAPIVersion"):   # paho-mqtt 2.x
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                                       clean_session=True)
        else:
            self._client = mqtt.Client(client_id=client_id, clean_session=True)
        if username:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._queues = {}
        self._connected = threading.Event()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if getattr(rc, "is_failure", rc != 0):
            print(f"MQTT connect to {self.host}:{self.port} refused: {rc}")
            return
        for topic in list(self._queues):
            client.subscribe(topic, qos=1)
        self._connected.set()

    def _on_message(self, client, userdata, msg):
        q = self._queues.get(msg.topic)
        if q is not None:
            q.put(msg.payload)

    def connect(self, timeout=10.0):
        self._client.connect(self.host, self.port, keepalive=60)
        self._client.loop_start()
        if not self._connected.wait(timeout):
            self.close()
            raise ConnectionError(f"no MQTT connection to {self.host}:{self.port} after {timeout:.0f} s")
        return self

    def subscribe(self, topic):
        q = self._queues.setdefault(topic, queue.Queue())
        if self._connected.is_set():
            self._client.subscribe(topic, qos=1)
        return q

    def publish(self, topic, payload):
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self._client.publish(topic, text, qos=1)

    def close(self):
        self._client.loop_stop()
        self._client.disconnect()


class LocalBus:
    """In-process stand-in for MqttLink (tests, dry runs): same subscribe /
    publish, delivered at once to every subscriber of the topic."""

    def __init__(self):
        self._queues = {}
        self._lock = threading.Lock()
        self.sent = []   # (topic, dict or text), in order

    def connect(self, timeout=None):
        return self

    def subscribe(self, topic):
        with self._lock:
            q = queue.Queue()
            self._queues.setdefault(topic, []).append(q)
            return q

    def publish(self, topic, payload):
        text = payload if isinstance(payload, str) else json.dumps(payload)
        with self._lock:
            self.sent.append((topic, payload))
            targets = list(self._queues.get(topic, []))
        for q in targets:
            q.put(text.encode("utf-8"))

    def close(self):
        pass
