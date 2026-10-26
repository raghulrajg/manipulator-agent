"""
mqtt_link.py
-------------
The MQTT side of the pick, shared by the vision PC (foundationpose_mqtt.py)
and the arm (pick_runner.py). Copy this file next to foundationpose_mqtt.py.

    arm    -> MQTT_READY_TOPIC  {"request_id", "joints", "stamp"}
              "arm is still at the detection pose, find the object"
              (a plain "ready" payload works too, e.g. from mosquitto_pub)
    vision -> MQTT_POSE_TOPIC   {"request_id", "T_camera_object" (4x4, m),
              "object", "origin", "symmetric", "frame": "camera", "diameter",
              "settle_s", "stamp"}
    arm    -> MQTT_STATUS_TOPIC {"request_id", "state", "detail", "stamp"}

    vision agent (or you, by hand) -> MQTT_COMMAND_TOPIC
              {"state": "start" | "stop" | "idle", "product": "tomato",
               "action": "pick" | "place", "position": [x, y, z, w]}
              position: the object (pick) or the spot to put it (place) in
              the wrist camera's frame, meters (x right, y down, z along
              the view, as FoundationPose); w: its turn about the vertical
              in degrees, ignored for a round object. See parse_command().
    arm    -> MQTT_STATUS_TOPIC {"state", "action", "product", "detail", "stamp"}
              for each command: accepted / refused / done / failed / stopped

Broker settings come from the same environment variables as the rest of
your MQTT stack: MQTT_BROKER_HOST (default localhost), MQTT_BROKER_PORT
(1883, or 8883 with TLS), MQTT_USERNAME, MQTT_PASSWORD; for EMQX Cloud or
an EMQX listener with TLS, MQTT_TLS=1 (on by default on port 8883) and
MQTT_CA_FILE (the broker's CA certificate, default: the system's); over
WebSocket, MQTT_TRANSPORT=websockets (EMQX: port 8083, or 8084 with TLS)
and MQTT_WS_PATH (default /mqtt). The topics can be changed with
MQTT_READY_TOPIC, MQTT_POSE_TOPIC, MQTT_STATUS_TOPIC and MQTT_COMMAND_TOPIC.

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
COMMAND_TOPIC = os.environ.get("MQTT_COMMAND_TOPIC", "manipulator/command")

SETTLE_SECONDS = 2.5   # object tracked and still this long before its pose is sent
SETTLE_MOVE = 0.01     # [m] a pose that moves more than this restarts the wait
RESEND_MAX_AGE = 10.0  # [s] a repeated ready gets the sent pose again only this soon after


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


# ------------------------------------------------------------ commands

COMMAND_STATES = ("start", "stop", "idle")
COMMAND_ACTIONS = ("pick", "place")
_STATE_SPELLINGS = {"idal": "idle", "idel": "idle"}


def _position(value):
    """[x, y, z(, w)], {"x", "y", "z"(, "w")} or "x,y,z(,w)" -> 4 floats (w 0 if left out)."""
    if isinstance(value, str):
        value = [v for v in value.replace(",", " ").split()]
    elif isinstance(value, dict):
        try:
            value = [value[k] for k in ("x", "y", "z")] + ([value["w"]] if "w" in value else [])
        except KeyError as e:
            raise MessageError(f"position needs x, y and z (missing {e})") from None
    if not isinstance(value, (list, tuple)) or len(value) not in (3, 4) or \
            any(isinstance(v, bool) for v in value):
        raise MessageError("position must be [x, y, z, w] (or x, y, z)")
    try:
        p = np.array([float(v) for v in value] + [0.0] * (4 - len(value)))
    except (TypeError, ValueError):
        raise MessageError("position must be numbers") from None
    if not np.all(np.isfinite(p)):
        raise MessageError("position must be finite numbers")
    if np.linalg.norm(p[:3]) > 2.0:
        raise MessageError(f"position {p[:3].tolist()} is more than 2 m from the camera: it is in meters, "
                           "not mm")
    if p[2] <= 0.0:
        raise MessageError(f"position z {p[2]} must be in front of the camera (z > 0, along the view)")
    if abs(p[3]) > 360:
        raise MessageError(f"w {p[3]} is a turn in degrees: -360..360")
    return p


def parse_command(payload):
    """A command from the vision agent -> {"state", "action", "product",
    "position": 4 floats or None, "request_id"}. Raises MessageError if it
    is not one. request_id is optional; the status replies carry it back.

    state   "start" runs the action, "stop" stops the arm (always works),
            "idle" does nothing but update the object (if position given)
    action  "pick": position is the object; "place": position is where to
            put the held object down (its x, y; it goes on the table)
    """
    text = payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else str(payload)
    if text.strip().strip('"').lower() == "stop":
        return {"state": "stop", "action": None, "product": "", "position": None, "request_id": None}
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        raise MessageError(f"not a JSON command: {text[:80]!r}") from None
    if not isinstance(msg, dict):
        raise MessageError(f"not a command: {text[:80]!r}")
    state = str(msg.get("state", "")).strip().lower()
    state = _STATE_SPELLINGS.get(state, state)
    if state not in COMMAND_STATES:
        raise MessageError(f"state must be start, stop or idle, got {msg.get('state')!r}")
    action = msg.get("action")
    action = None if action in (None, "") else str(action).strip().lower()
    if action is not None and action not in COMMAND_ACTIONS:
        raise MessageError(f"action must be pick or place, got {msg.get('action')!r}")
    if state == "start" and action is None:
        raise MessageError("state start needs an action: pick or place")
    product = msg.get("product") or ""
    if not isinstance(product, str):
        raise MessageError("product must be a string")
    position = None if msg.get("position") in (None, "", []) else _position(msg["position"])
    if state == "start" and action == "place" and position is None:
        raise MessageError("place needs a position: where to put the object")
    _check_request_id(msg, text)   # optional, echoed in the status replies
    return {"state": state, "action": action, "product": product[:64], "position": position,
            "request_id": msg["request_id"]}


def command_message(state, action=None, position=None, product=""):
    """The command as the vision agent sends it (for mqtt_send.py and tests)."""
    msg = {"state": state, "product": product}
    if action is not None:
        msg["action"] = action
    if position is not None:
        msg["position"] = [round(float(v), 6) for v in position]
    return msg


def camera_point_pose(position):
    """[x, y, z, w] in the camera frame -> 4x4 T_camera_object: at x, y, z;
    turned w degrees about the camera's view axis. For a top-down view
    that's a turn about the vertical; a round object ignores it."""
    x, y, z, w = (float(v) for v in position)
    c, s = np.cos(np.radians(w)), np.sin(np.radians(w))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [x, y, z]
    return T


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

    def forget(self, request_id):
        """Treat request_id as never answered (its pose is too old to send
        again), so arm() takes it for a fresh detection."""
        self.answered.discard(request_id)

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

    def __init__(self, client_id, host=None, port=None, username=None, password=None,
                 tls=None, ca_file=None, transport=None, ws_path=None):
        """Works with any MQTT 3.1.1 broker: EMQX, Mosquitto, ... Each client
        needs its own client_id (EMQX drops the older of two with the same
        one), so callers add a random suffix."""
        import paho.mqtt.client as mqtt   # lazy, so the rest imports without paho
        env = os.environ.get
        self.host = host or env("MQTT_BROKER_HOST", "localhost")
        self.port = int(port or env("MQTT_BROKER_PORT", 1883))
        username = username if username is not None else env("MQTT_USERNAME")
        password = password if password is not None else env("MQTT_PASSWORD")
        if tls is None:
            flag = env("MQTT_TLS", "").strip().lower()
            tls = flag in ("1", "true", "yes", "on") if flag else self.port in (8883, 8084)
        self.tls = bool(tls)
        transport = transport or env("MQTT_TRANSPORT", "tcp")
        if transport not in ("tcp", "websockets"):
            raise ValueError("MQTT_TRANSPORT must be tcp or websockets")
        kw = dict(client_id=client_id, clean_session=True, transport=transport)
        if hasattr(mqtt, "CallbackAPIVersion"):   # paho-mqtt 2.x
            self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, **kw)
        else:
            self._client = mqtt.Client(**kw)
        if transport == "websockets":
            self._client.ws_set_options(path=ws_path or env("MQTT_WS_PATH", "/mqtt"))
        if self.tls:
            self._client.tls_set(ca_certs=ca_file or env("MQTT_CA_FILE") or None)
        if username:
            self._client.username_pw_set(username, password)
        self.refused = None   # the broker's reason, if it refused the connection
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._queues = {}
        self._connected = threading.Event()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if getattr(rc, "is_failure", rc != 0):
            self.refused = str(rc)
            print(f"MQTT connect to {self.host}:{self.port} refused: {rc}"
                  + (" (check MQTT_USERNAME / MQTT_PASSWORD and the broker's authentication / ACL)"
                     if "auth" in str(rc).lower() or "password" in str(rc).lower() else ""))
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
            why = f": the broker said {self.refused}" if self.refused else ""
            raise ConnectionError(f"no MQTT connection to {self.host}:{self.port}"
                                  f"{' (TLS)' if self.tls else ''} after {timeout:.0f} s{why}")
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
