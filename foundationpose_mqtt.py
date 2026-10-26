"""
foundationpose_mqtt.py
-----------------------
Tomato 6D pose (RealSense + YOLO + SAM + FoundationPose) for the arm, over
MQTT. The vision pipeline is yours unchanged; MQTT decides when it runs:

  1. idle until a "ready" message on MQTT_READY_TOPIC (pick_runner.py sends
     it once the arm is at the detection pose and holding still)
  2. find the tomato: YOLO -> SAM -> FoundationPose register, then track
  3. once it has been tracked for 2.5 s without moving more than 1 cm
     (mqtt_link.SETTLE_SECONDS / SETTLE_MOVE), publish the pose once on
     MQTT_POSE_TOPIC: T_camera_object (camera optical frame, meters, origin
     at the tomato's center) plus the mesh diameter
  4. back to idle -- the arm moves the camera now -- until the next ready

Put this file and mqtt_link.py in the folder with FoundationPose/,
meshes/, the SAM checkpoint, precise_6dof_pose.py and the camera modules.
Broker: MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_USERNAME, MQTT_PASSWORD.

Keys: SPACE = look now without a ready message, R = force re-registration,
ESC = exit.
"""

import queue
import sys
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as Rot

from mqtt_link import (
    COMMAND_TOPIC, POSE_TOPIC, READY_TOPIC, RESEND_MAX_AGE, MessageError, MqttLink, PoseGate,
    command_message, matrix_quaternion, parse_ready, pose_message,
)

# Vision only: no manipulator / IK code in this version. The pose goes to
# the arm over MQTT (pick_runner.py turns it into joint angles).


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FP_PATH = BASE_DIR / "FoundationPose"

MESH_PATH = (
    BASE_DIR
    / "meshes"
    / "tomato"
    / "tomato.obj"
)

SAM_CHECKPOINT = (
    BASE_DIR
    / "sam_vit_b_01ec64.pth"
)

DEBUG_DIR = BASE_DIR / "fp_debug"


# ============================================================
# FOUNDATIONPOSE IMPORT
# ============================================================

sys.path.insert(0, str(FP_PATH))

from estimater import (
    FoundationPose,
    ScorePredictor,
    PoseRefinePredictor,
)

import nvdiffrast.torch as dr


# ============================================================
# CAMERA IMPORT
# ============================================================

# Default: frames come from router.py (which reads rs_sender.py on the laptop).
# Run with --local-camera to use a RealSense plugged into this machine.
RS_HOST = "127.0.0.1"
RS_PORT = 5601          # router.py FoundationPose port (5577 = laptop directly)

if "--local-camera" in sys.argv:
    from realsense_stream import CameraStreamer
else:
    import functools
    from remote_camera import CameraStreamer
    CameraStreamer = functools.partial(CameraStreamer, host=RS_HOST, port=RS_PORT)


# ============================================================
# DETECTOR / SAM / VISUALIZATION
# ============================================================

from precise_6dof_pose import (
    Detector,
    Segmenter,
    draw_pose_axes,
    Pose6D,
)


# ============================================================
# FOUNDATIONPOSE SETUP
# ============================================================

# Target size for the mesh FoundationPose renders. 252 hypotheses are rendered
# in one batch, so GPU memory scales with vertex count * 252.
MAX_FACES = 20000
MAX_TEXTURE_SIZE = 1024


def load_and_clean_mesh(mesh_path):

    # force="mesh" merges an OBJ that loads as a Scene into one Trimesh
    mesh = trimesh.load(
        str(mesh_path),
        force="mesh",
        process=False
    )

    print(f"Mesh vertices: {len(mesh.vertices)}")
    print(f"Mesh faces:    {len(mesh.faces)}")
    print(f"Mesh extents (raw): {mesh.extents}")
    print(f"Mesh center  (raw): {mesh.bounds.mean(axis=0)}")

    # --------------------------------------------------------
    # UNITS: FoundationPose expects meters. A tomato is ~0.05-0.10 m.
    # --------------------------------------------------------

    max_extent = float(np.max(mesh.extents))

    if max_extent > 10.0:
        print("Mesh looks like millimeters -> scaling by 0.001")
        mesh.apply_scale(0.001)
    elif max_extent > 1.0:
        print("Mesh looks like centimeters -> scaling by 0.01")
        mesh.apply_scale(0.01)

    # --------------------------------------------------------
    # CENTER the mesh at its bounding-box center
    # --------------------------------------------------------

    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    # --------------------------------------------------------
    # SIMPLIFY high-poly scans (main cause of cudaMalloc error 2)
    # --------------------------------------------------------

    if len(mesh.faces) > MAX_FACES:

        print(f"Simplifying {len(mesh.faces)} -> {MAX_FACES} faces...")

        # Decimation drops UVs, so color the vertices instead
        try:
            vertex_colors = mesh.visual.to_color().vertex_colors
        except Exception:
            vertex_colors = None

        try:
            # needs: pip install fast-simplification
            simplified = mesh.simplify_quadric_decimation(
                face_count=MAX_FACES
            )
        except Exception as e:
            print(f"trimesh decimation failed ({e}), trying open3d...")
            import open3d as o3d
            o3d_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh.vertices),
                o3d.utility.Vector3iVector(mesh.faces)
            )
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(MAX_FACES)
            simplified = trimesh.Trimesh(
                np.asarray(o3d_mesh.vertices),
                np.asarray(o3d_mesh.triangles),
                process=False
            )

        if vertex_colors is not None:
            # Copy color from the nearest original vertex
            from scipy.spatial import cKDTree
            _, idx = cKDTree(mesh.vertices).query(simplified.vertices)
            simplified.visual = trimesh.visual.ColorVisuals(
                simplified,
                vertex_colors=vertex_colors[idx]
            )

        mesh = simplified

    # --------------------------------------------------------
    # SHRINK huge textures (an 8K texture alone is ~200 MB on GPU)
    # --------------------------------------------------------

    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "image", None)

    if image is not None and max(image.size) > MAX_TEXTURE_SIZE:
        print(f"Resizing texture {image.size} -> {MAX_TEXTURE_SIZE}px")
        image = image.copy()
        image.thumbnail((MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE))
        mesh.visual.material.image = image

    print(f"Mesh vertices (final): {len(mesh.vertices)}")
    print(f"Mesh faces    (final): {len(mesh.faces)}")
    print(f"Mesh extents  (final, m): {mesh.extents}")

    return mesh


def setup_foundation_pose(mesh_path):

    print(f"Loading 3D mesh: {mesh_path}...")

    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Mesh not found:\n{mesh_path}"
        )

    mesh = load_and_clean_mesh(mesh_path)

    # ========================================================
    # UV DATA FOR FOUNDATIONPOSE
    # ========================================================

    # A textured mesh keeps its UVs. A vertex-colored mesh (e.g. after
    # simplification) must NOT get dummy UVs, or FoundationPose will
    # treat it as textured and sample a missing texture.

    if isinstance(mesh.visual, trimesh.visual.TextureVisuals):

        uv = mesh.visual.uv

        if (
            uv is None
            or np.asarray(uv).shape != (len(mesh.vertices), 2)
            or not np.isfinite(uv).all()
        ):
            print("Invalid UVs, converting to vertex colors...")
            mesh.visual = mesh.visual.to_color()
        else:
            mesh.visual.uv = np.asarray(uv, dtype=np.float32)

    print("Visual kind:", mesh.visual.kind)

    # ========================================================
    # FOUNDATIONPOSE OBJECT
    # ========================================================

    print(
        "Initializing ScorePredictor..."
    )

    scorer = ScorePredictor()

    print(
        "Initializing PoseRefinePredictor..."
    )

    refiner = PoseRefinePredictor()

    print(
        "Initializing CUDA rasterizer..."
    )

    glctx = dr.RasterizeCudaContext()

    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(DEBUG_DIR),
        debug=0,
        glctx=glctx
    )

    print(f"FoundationPose diameter: {est.diameter:.4f} m")

    return est, mesh


# ============================================================
# DEPTH VALIDATION
# ============================================================

def validate_mask_depth(mask, depth):

    if mask is None:
        return False, 0, 0.0

    if depth is None:
        return False, 0, 0.0

    mask_bool = mask.astype(bool)

    mask_pixels = int(
        mask_bool.sum()
    )

    if mask_pixels < 500:

        return (
            False,
            mask_pixels,
            0.0
        )

    valid_depth = (
        mask_bool
        & np.isfinite(depth)
        & (depth >= 0.15)
        & (depth <= 3.0)
    )

    valid_pixels = int(
        valid_depth.sum()
    )

    depth_ratio = (
        valid_pixels
        / max(mask_pixels, 1)
    )

    valid = (
        valid_pixels >= 300
        and depth_ratio >= 0.20
    )

    return (
        valid,
        valid_pixels,
        depth_ratio,
    )


# ============================================================
# BBOX IoU
# ============================================================

# ============================================================
# POSE SMOOTHING
# ============================================================

# A tomato is close to a sphere, so FoundationPose cannot tell its
# roll/pitch/yaw from shape alone; the rotation jumps between frames
# that all look identical. Only the position is reliable.
# LOCK_ROTATION=True keeps the object axes parallel to the camera axes
# so the IK target orientation is stable.
LOCK_ROTATION = True

# TWIN_STREAM=1: also send every tracked pose (position + quaternion, camera
# frame) to the digital twin as an "idle" command, TWIN_STREAM_HZ a second,
# so the object in the twin follows the real one. Nothing moves the arm.
TWIN_STREAM = os.environ.get("TWIN_STREAM", "0").strip().lower() in ("1", "true", "yes", "on")
TWIN_STREAM_HZ = float(os.environ.get("TWIN_STREAM_HZ", "5"))
POS_ALPHA = 0.5   # 0..1, lower = smoother but more lag
ROT_ALPHA = 0.3
# Moves larger than this are real motion: follow them immediately
# instead of easing in, so the frame does not lag behind the tomato.
SNAP_DIST = 0.008  # meters

# Re-anchor tracking when the pose center drifts out of the YOLO box.
REANCHOR_MARGIN = 0.15  # fraction of bbox size allowed outside the box
MIN_DEPTH_RATIO = 0.3   # below this, depth is too sparse to trust


class PoseSmoother:

    def __init__(self):
        self.pose = None

    def reset(self):
        self.pose = None

    def update(self, pose_matrix):

        if pose_matrix is None:
            return None

        new = np.asarray(pose_matrix, dtype=np.float64).copy()

        if LOCK_ROTATION:
            new[:3, :3] = np.eye(3)

        if self.pose is None:
            self.pose = new
            return self.pose.copy()

        out = np.eye(4)

        jump = np.linalg.norm(new[:3, 3] - self.pose[:3, 3])

        if jump > SNAP_DIST:
            # Real motion: follow immediately
            out[:3, 3] = new[:3, 3]
        else:
            # Small jitter: exponential moving average
            out[:3, 3] = (
                POS_ALPHA * new[:3, 3]
                + (1.0 - POS_ALPHA) * self.pose[:3, 3]
            )

        # Slerp-style blend on rotation
        r_old = Rot.from_matrix(self.pose[:3, :3])
        r_new = Rot.from_matrix(new[:3, :3])
        delta = (r_old.inv() * r_new).as_rotvec()
        out[:3, :3] = (
            r_old * Rot.from_rotvec(ROT_ALPHA * delta)
        ).as_matrix()

        self.pose = out
        return out.copy()


def depth_hole_ratio(pose_matrix, depth, K, diameter):
    """Fraction of missing depth pixels around the tracked tomato."""

    t = pose_matrix[:3, 3]

    if t[2] <= 0:
        return 1.0

    u = int(K[0, 0] * t[0] / t[2] + K[0, 2])
    v = int(K[1, 1] * t[1] / t[2] + K[1, 2])
    r = max(3, int(0.3 * K[0, 0] * diameter / t[2]))

    h, w = depth.shape[:2]
    patch = depth[
        max(0, v - r):min(h, v + r),
        max(0, u - r):min(w, u + r)
    ]

    if patch.size == 0:
        return 1.0

    return float(np.mean(~(patch > 0)))


def reanchor_to_bbox(est, pose_matrix, bbox, depth, K):
    """
    Check whether the tracked center still lies inside the YOLO box.
    If it drifted out, move FoundationPose's internal pose to the box
    center at the measured depth so tracking recovers next frame.
    Returns True when a re-anchor happened.
    """

    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1

    if w <= 0 or h <= 0:
        return False

    t = pose_matrix[:3, 3]

    if t[2] <= 0:
        u, v = -1.0, -1.0
    else:
        u = K[0, 0] * t[0] / t[2] + K[0, 2]
        v = K[1, 1] * t[1] / t[2] + K[1, 2]

    mx, my = REANCHOR_MARGIN * w, REANCHOR_MARGIN * h
    inside = (x1 - mx <= u <= x2 + mx) and (y1 - my <= v <= y2 + my)

    # Depth at the center of the box (inner 40%)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    rw, rh = max(1, int(0.2 * w)), max(1, int(0.2 * h))
    patch = depth[
        max(0, cy - rh):cy + rh,
        max(0, cx - rw):cx + rw
    ]
    valid = patch[(patch > 0.05) & np.isfinite(patch)]

    if valid.size < MIN_DEPTH_RATIO * patch.size:
        return False

    # Surface depth + radius = tomato center depth
    z = float(np.median(valid)) + est.diameter / 2.0
    depth_off = abs(t[2] - z) > est.diameter

    if inside and not depth_off:
        return False

    new_t = np.array([
        (cx - K[0, 2]) * z / K[0, 0],
        (cy - K[1, 2]) * z / K[1, 1],
        z
    ])

    pose_last = est.pose_last.clone().reshape(4, 4)
    pose_last[:3, 3] = torch.as_tensor(
        new_t,
        dtype=pose_last.dtype,
        device=pose_last.device
    )
    est.pose_last = pose_last.reshape(est.pose_last.shape)

    # Update the displayed pose too, so it jumps back this frame
    pose_matrix[:3, 3] = new_t

    print(
        f"Re-anchored tracking: center was "
        f"{'outside box' if not inside else 'at wrong depth'}"
    )

    return True


def bbox_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    iw = max(
        0,
        ix2 - ix1
    )

    ih = max(
        0,
        iy2 - iy1
    )

    intersection = (
        iw * ih
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# FOUNDATIONPOSE MATRIX -> Pose6D
# ============================================================

def foundation_pose_to_6d(pose_matrix):

    if pose_matrix is None:
        return None

    pose_matrix = np.asarray(
        pose_matrix,
        dtype=np.float64
    )

    if pose_matrix.shape != (4, 4):

        print(
            "Invalid pose matrix shape:",
            pose_matrix.shape
        )

        return None

    if not np.all(
        np.isfinite(pose_matrix)
    ):

        print(
            "Pose matrix contains NaN/Inf."
        )

        return None

    # --------------------------------------------------------
    # Extract translation
    # --------------------------------------------------------

    position = pose_matrix[
        :3,
        3
    ].copy()

    # --------------------------------------------------------
    # Extract rotation
    # --------------------------------------------------------

    rotation = pose_matrix[
        :3,
        :3
    ].copy()

    # --------------------------------------------------------
    # Validate rotation
    # --------------------------------------------------------

    det = np.linalg.det(
        rotation
    )

    if not np.isfinite(det):

        print(
            "Invalid rotation determinant."
        )

        return None

    # FoundationPose should already return a valid rotation.
    # SVD projection makes the rotation numerically safe.
    U, _, Vt = np.linalg.svd(
        rotation
    )

    rotation = U @ Vt

    if np.linalg.det(rotation) < 0:

        U[:, -1] *= -1

        rotation = U @ Vt

    # --------------------------------------------------------
    # Quaternion
    #
    # scipy returns:
    # [x, y, z, w]
    # --------------------------------------------------------

    quat = Rot.from_matrix(
        rotation
    ).as_quat()

    # --------------------------------------------------------
    # Euler angles
    #
    # XYZ convention:
    # roll  = X
    # pitch = Y
    # yaw   = Z
    # --------------------------------------------------------

    euler = Rot.from_matrix(
        rotation
    ).as_euler(
        "xyz",
        degrees=True
    )

    # --------------------------------------------------------
    # Return Pose6D
    # --------------------------------------------------------

    return Pose6D(
        position=position,
        rotation=rotation,
        quaternion=quat,
        euler_xyz_deg=euler,
        n_points=0,
        confidence=None
    )


# ============================================================
# PRINT 6D POSE
# ============================================================

def print_6d_pose(pose_data):

    if pose_data is None:
        return

    position = pose_data.position
    rotation = pose_data.rotation
    quaternion = pose_data.quaternion
    euler = pose_data.euler_xyz_deg

    print()
    print(
        "============================================================"
    )

    print(
        "FOUNDATIONPOSE 6D POSE"
    )

    print(
        "------------------------------------------------------------"
    )

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    print(
        "Position (camera frame):"
    )

    print(
        f"  X = {position[0]: .4f} m"
    )

    print(
        f"  Y = {position[1]: .4f} m"
    )

    print(
        f"  Z = {position[2]: .4f} m"
    )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distance = np.linalg.norm(
        position
    )

    print(
        f"  Distance = {distance:.4f} m"
    )

    # --------------------------------------------------------
    # Rotation matrix
    # --------------------------------------------------------

    print()
    print(
        "Rotation Matrix:"
    )

    for row in rotation:

        print(
            " ",
            " ".join(
                f"{value: .5f}"
                for value in row
            )
        )

    # --------------------------------------------------------
    # Quaternion
    # --------------------------------------------------------

    print()
    print(
        "Quaternion (xyzw):"
    )

    print(
        f"  X = {quaternion[0]: .5f}"
    )

    print(
        f"  Y = {quaternion[1]: .5f}"
    )

    print(
        f"  Z = {quaternion[2]: .5f}"
    )

    print(
        f"  W = {quaternion[3]: .5f}"
    )

    # --------------------------------------------------------
    # Euler
    # --------------------------------------------------------

    print()
    print(
        "Euler XYZ:"
    )

    print(
        f"  Roll  = {euler[0]: .2f} deg"
    )

    print(
        f"  Pitch = {euler[1]: .2f} deg"
    )

    print(
        f"  Yaw   = {euler[2]: .2f} deg"
    )

    print(
        "============================================================"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "tomato - RealSense + YOLO + SAM + FoundationPose"
    )

    print(
        "6D Pose Estimation"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # MQTT
    # ========================================================

    print(
        "\n[0] Connecting to MQTT..."
    )

    link = MqttLink(
        # EMQX drops the older of two clients with one ID: keep each run's own
        client_id=f"foundationpose_vision_{uuid.uuid4().hex[:6]}"
    )

    ready_queue = link.subscribe(
        READY_TOPIC
    )

    link.connect()

    gate = PoseGate()

    # request_id -> (pose message sent, when), to send again if the arm asks again
    sent_poses = {}

    print(
        f"MQTT ready: {link.host}:{link.port}, "
        f"ready on '{READY_TOPIC}', poses to '{POSE_TOPIC}'."
    )

    last_streamed = -float("inf")

    if TWIN_STREAM:
        print(
            f"Streaming the tracked pose to the twin on '{COMMAND_TOPIC}' "
            f"({TWIN_STREAM_HZ:g} Hz, state idle)."
        )

    # ========================================================
    # CAMERA
    # ========================================================

    print(
        "\n[1] Starting RealSense..."
    )

    cam = CameraStreamer(
        width=640,
        height=480,
        fps=30
    )

    print(
        "RealSense ready."
    )

    print(
        "\nCamera K:"
    )

    print(
        cam.K
    )

    print(
        "\nDepth scale:"
    )

    print(
        cam.depth_scale
    )

    # ========================================================
    # CAMERA DISTORTION
    # ========================================================

    camera_dist = getattr(
        cam,
        "dist",
        None
    )

    if camera_dist is None:

        camera_dist = np.zeros(
            5,
            dtype=np.float64
        )

        print(
            "\nCamera distortion coefficients "
            "not available."
        )

        print(
            "Using zeros."
        )

    else:

        camera_dist = np.asarray(
            camera_dist
        )

        print(
            "\nCamera distortion:"
        )

        print(
            camera_dist
        )

    # ========================================================
    # FOUNDATIONPOSE
    # ========================================================

    print(
        "\n[2] Initializing FoundationPose..."
    )

    est, mesh = setup_foundation_pose(
        MESH_PATH
    )

    print(
        "FoundationPose ready."
    )

    # ========================================================
    # YOLO
    # ========================================================

    print(
        "\n[3] Initializing YOLO..."
    )

    detector = Detector(
        conf=0.25
    )

    detector.set_prompt(
        "tomato"
    )

    print(
        "YOLO ready."
    )

    # ========================================================
    # SAM
    # ========================================================

    print(
        "\n[4] Initializing SAM..."
    )

    segmenter = Segmenter(
        str(SAM_CHECKPOINT),
        "vit_b",
        device="cuda"
    )

    print(
        "SAM ready."
    )

    # ========================================================
    # STATE
    # ========================================================

    is_registered = False

    smoother = PoseSmoother()

    pose_matrix = None

    frame_count = 0

    lost_counter = 0

    stable_counter = 0

    previous_bbox = None

    candidate_mask = None

    candidate_bbox = None

    # ========================================================
    # PARAMETERS
    # ========================================================

    PRINT_EVERY = 10

    CHECK_EVERY = 2

    LOST_THRESHOLD = 5

    STABLE_FRAMES_REQUIRED = 5

    MIN_BBOX_IOU = 0.70

    # ========================================================
    # START
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Pipeline Ready"
    )

    print(
        f"Waiting for 'ready' on {READY_TOPIC}."
    )

    print(
        "SPACE = look now without a ready message"
    )

    print(
        "R = force re-registration"
    )

    print(
        "ESC = exit"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    try:

        while True:

            # =================================================
            # CAMERA
            # =================================================

            color_bgr, depth_meters = (
                cam.get_frames()
            )

            if color_bgr is None:

                continue

            if depth_meters is None:

                continue

            frame_count += 1

            # -------------------------------------------------
            # RGB conversion
            # -------------------------------------------------

            color_rgb = cv2.cvtColor(
                color_bgr,
                cv2.COLOR_BGR2RGB
            )

            # -------------------------------------------------
            # Display image
            # -------------------------------------------------

            display = color_bgr.copy()

            # =================================================
            # MQTT: READY -> LOOK FOR THE TOMATO
            # =================================================

            while True:

                try:
                    payload = ready_queue.get_nowait()
                except queue.Empty:
                    break

                try:
                    ready = parse_ready(payload)
                except MessageError as e:
                    print(f"\nIgnoring message on {READY_TOPIC}: {e}")
                    continue

                rid = ready["request_id"]

                if (
                    rid in sent_poses
                    and time.monotonic() - sent_poses[rid][1] > RESEND_MAX_AGE
                ):
                    # Too old to trust (the tomato may have moved): look
                    # again, the arm is still at its detection pose
                    del sent_poses[rid]
                    gate.forget(rid)

                if gate.arm(rid):

                    print(
                        f"\nReady received (request "
                        f"{ready['request_id']}): looking for the tomato."
                    )

                    # Fresh detection: the camera has moved since the last one
                    is_registered = False
                    pose_matrix = None
                    lost_counter = 0
                    stable_counter = 0
                    previous_bbox = None

                elif rid in sent_poses:

                    # The arm is still waiting: our pose did not reach it
                    link.publish(
                        POSE_TOPIC,
                        sent_poses[rid][0]
                    )

                    print(
                        f"\nRepeated ready (request "
                        f"{ready['request_id']}): pose sent again."
                    )

            # =================================================
            # IDLE: NO READY YET
            # =================================================

            if not gate.armed:

                cv2.putText(
                    display,
                    "Waiting for ready (MQTT)...",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            # =================================================
            # NOT REGISTERED
            # =================================================

            elif not is_registered:

                # Not tracked: the 2.5 s wait starts again once it is
                gate.restart()

                # ------------------------------------------------
                # YOLO detection
                # ------------------------------------------------

                dets = detector.detect(
                    color_bgr
                )

                if not dets:

                    stable_counter = 0

                    previous_bbox = None

                    cv2.putText(
                        display,
                        "Waiting for tomato...",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                else:

                    # ------------------------------------------------
                    # Best detection
                    # ------------------------------------------------

                    best = dets[0]

                    bbox = np.array(
                        best["bbox"],
                        dtype=np.int32
                    )

                    # ------------------------------------------------
                    # BBOX stability
                    # ------------------------------------------------

                    stable = True

                    if previous_bbox is not None:

                        iou = bbox_iou(
                            previous_bbox,
                            bbox
                        )

                        stable = (
                            iou >= MIN_BBOX_IOU
                        )

                    else:

                        iou = 0.0

                    if stable:

                        stable_counter += 1

                    else:

                        stable_counter = 1

                    previous_bbox = bbox.copy()

                    # ------------------------------------------------
                    # Draw bounding box
                    # ------------------------------------------------

                    x1, y1, x2, y2 = bbox

                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        display,
                        f"Detection {best['conf']:.2f}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    # ------------------------------------------------
                    # Stability information
                    # ------------------------------------------------

                    cv2.putText(
                        display,
                        f"Stable: "
                        f"{stable_counter}/"
                        f"{STABLE_FRAMES_REQUIRED}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    # ------------------------------------------------
                    # IoU
                    # ------------------------------------------------

                    cv2.putText(
                        display,
                        f"IoU: {iou:.2f}",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    # =================================================
                    # SAM
                    # =================================================

                    if (
                        stable_counter
                        >= STABLE_FRAMES_REQUIRED
                    ):

                        mask = segmenter.segment(
                            color_bgr,
                            bbox
                        )

                        (
                            valid_mask,
                            valid_pixels,
                            depth_ratio
                        ) = validate_mask_depth(
                            mask,
                            depth_meters
                        )

                        candidate_mask = mask

                        candidate_bbox = (
                            bbox.copy()
                        )

                        # ------------------------------------------------
                        # Mask overlay
                        # ------------------------------------------------

                        mask_overlay = (
                            np.zeros_like(display)
                        )

                        if mask is not None:

                            mask_overlay[mask] = (
                                0,
                                255,
                                0
                            )

                            display = (
                                cv2.addWeighted(
                                    display,
                                    0.75,
                                    mask_overlay,
                                    0.25,
                                    0
                                )
                            )

                        # ------------------------------------------------
                        # Depth information
                        # ------------------------------------------------

                        cv2.putText(
                            display,
                            f"Depth: {valid_pixels} "
                            f"ratio={depth_ratio:.2f}",
                            (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (
                                (0, 255, 0)
                                if valid_mask
                                else
                                (0, 0, 255)
                            ),
                            2
                        )

                        # =================================================
                        # FOUNDATIONPOSE REGISTRATION
                        # =================================================

                        if valid_mask:

                            print(
                                "\nStable tomato detected."
                            )

                            print(
                                f"Valid depth pixels: "
                                f"{valid_pixels}"
                            )

                            print(
                                f"Mask depth ratio: "
                                f"{depth_ratio:.3f}"
                            )

                            print(
                                "Registering FoundationPose..."
                            )

                            try:

                                # Free memory cached by YOLO/SAM before
                                # FoundationPose renders 252 hypotheses
                                torch.cuda.empty_cache()

                                free_b, total_b = torch.cuda.mem_get_info()
                                print(
                                    f"GPU free before register: "
                                    f"{free_b / 1e9:.2f} / {total_b / 1e9:.2f} GB"
                                )

                                pose_matrix = (
                                    est.register(
                                        K=cam.K,
                                        rgb=color_rgb,
                                        depth=depth_meters,
                                        ob_mask=mask,
                                        iteration=5
                                    )
                                )

                            except Exception as e:

                                print(
                                    "\nRegistration error:"
                                )

                                print(
                                    str(e)
                                )

                                pose_matrix = None

                            # =================================================
                            # CHECK REGISTRATION RESULT
                            # =================================================

                            if (
                                pose_matrix is not None
                                and np.all(
                                    np.isfinite(
                                        pose_matrix
                                    )
                                )
                            ):

                                position = (
                                    pose_matrix[:3, 3]
                                )

                                distance = np.linalg.norm(
                                    position
                                )

                                print(
                                    f"Initial position: "
                                    f"{position}"
                                )

                                print(
                                    f"Initial distance: "
                                    f"{distance:.3f} m"
                                )

                                # ------------------------------------------------
                                # Convert to 6D
                                # ------------------------------------------------

                                pose_data = (
                                    foundation_pose_to_6d(
                                        pose_matrix
                                    )
                                )

                                if pose_data is not None:

                                    print_6d_pose(
                                        pose_data
                                    )

                                # ------------------------------------------------
                                # Sanity check
                                # ------------------------------------------------

                                if (
                                    0.10
                                    < distance
                                    < 3.0
                                    and pose_data
                                    is not None
                                ):

                                    is_registered = (
                                        True
                                    )

                                    lost_counter = 0

                                    smoother.reset()

                                    pose_matrix = smoother.update(
                                        pose_matrix
                                    )

                                    stable_counter = 0

                                    previous_bbox = (
                                        None
                                    )

                                    print(
                                        "\nFoundationPose "
                                        "registration successful."
                                    )

                                    print(
                                        "6D tracking locked."
                                    )

                                else:

                                    print(
                                        "Registration returned "
                                        "an implausible pose."
                                    )

                                    pose_matrix = (
                                        None
                                    )

                                    stable_counter = (
                                        0
                                    )

                            else:

                                print(
                                    "Invalid pose returned."
                                )

                                pose_matrix = (
                                    None
                                )

                                stable_counter = (
                                    0
                                )

                # ------------------------------------------------
                # Registration status
                # ------------------------------------------------

                cv2.putText(
                    display,
                    "Press R to reset",
                    (
                        20,
                        display.shape[0] - 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            # =================================================
            # REGISTERED -> TRACKING
            # =================================================

            else:

                # ------------------------------------------------
                # FoundationPose tracking
                # ------------------------------------------------

                # Too close: RealSense returns no depth below its
                # minimum range (~0.28 m on a D435), and tracking
                # drifts on empty depth. Hold the last pose instead.
                hold_pose = False

                if pose_matrix is not None:
                    hole_ratio = depth_hole_ratio(
                        pose_matrix,
                        depth_meters,
                        np.asarray(cam.K),
                        est.diameter
                    )
                    hold_pose = hole_ratio > 0.5

                if hold_pose:
                    cv2.putText(
                        display,
                        "Too close to camera - holding last pose",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                try:

                    if not hold_pose:

                        pose_matrix = (
                            est.track_one(
                                rgb=color_rgb,
                                depth=depth_meters,
                                K=cam.K,
                                iteration=4
                            )
                        )

                        pose_matrix = smoother.update(
                            pose_matrix
                        )

                except Exception as e:

                    print(
                        f"\nTracking error: {e}"
                    )

                    is_registered = False

                    pose_matrix = None

                    lost_counter = 0

                    stable_counter = 0

                    previous_bbox = None

                    continue

                # =================================================
                # VALIDATE TRACKING POSE
                # =================================================

                pose_data = (
                    foundation_pose_to_6d(
                        pose_matrix
                    )
                )

                if pose_data is None:

                    print(
                        "\nInvalid tracking pose."
                    )

                    is_registered = False

                    pose_matrix = None

                    continue

                # -------------------------------------------------
                # Extract 6D data
                # -------------------------------------------------

                position = (
                    pose_data.position
                )

                rotation = (
                    pose_data.rotation
                )

                quat = (
                    pose_data.quaternion
                )

                euler = (
                    pose_data.euler_xyz_deg
                )

                distance = np.linalg.norm(
                    position
                )

                # =================================================
                # PERIODIC DETECTOR VERIFICATION
                # =================================================

                if (
                    frame_count
                    % CHECK_EVERY
                    == 0
                ):

                    dets = detector.detect(
                        color_bgr
                    )

                    if not dets:

                        lost_counter += 1

                        cv2.putText(
                            display,
                            f"Object not detected "
                            f"({lost_counter}/"
                            f"{LOST_THRESHOLD})",
                            (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2
                        )

                    else:

                        lost_counter = 0

                        bbox = np.array(
                            dets[0]["bbox"],
                            dtype=np.int32
                        )

                        if reanchor_to_bbox(
                            est,
                            pose_matrix,
                            bbox,
                            depth_meters,
                            np.asarray(cam.K)
                        ):
                            smoother.reset()

                        x1, y1, x2, y2 = (
                            bbox
                        )

                        cv2.rectangle(
                            display,
                            (x1, y1),
                            (x2, y2),
                            (255, 255, 0),
                            2
                        )

                        cv2.putText(
                            display,
                            f"YOLO {dets[0]['conf']:.2f}",
                            (
                                x1,
                                max(
                                    25,
                                    y1 - 10
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            2
                        )

                    # ------------------------------------------------
                    # Object lost
                    # ------------------------------------------------

                    if (
                        lost_counter
                        >= LOST_THRESHOLD
                    ):

                        print(
                            "\ntomato lost."
                        )

                        print(
                            "Resetting registration."
                        )

                        is_registered = (
                            False
                        )

                        pose_matrix = (
                            None
                        )

                        lost_counter = (
                            0
                        )

                        stable_counter = (
                            0
                        )

                        previous_bbox = (
                            None
                        )

                        continue

                # =================================================
                # MQTT: SEND THE POSE ONCE IT HAS HELD STILL
                # =================================================

                # A held pose (no depth) is not live tracking: don't count it
                T_send = gate.update(
                    None if hold_pose else pose_matrix,
                    time.monotonic()
                )

                # The live pose to the twin (TWIN_STREAM=1): its object follows
                if (
                    TWIN_STREAM
                    and pose_matrix is not None
                    and not hold_pose
                    and time.monotonic() - last_streamed >= 1.0 / TWIN_STREAM_HZ
                ):
                    last_streamed = time.monotonic()
                    T_live = np.asarray(pose_matrix, dtype=float)
                    stream = command_message(
                        "idle",
                        position=T_live[:3, 3],
                        product="tomato",
                        orientation=matrix_quaternion(T_live[:3, :3]),
                    )
                    # locked rotation = no real orientation to grip by
                    stream["symmetric"] = bool(LOCK_ROTATION)
                    link.publish(COMMAND_TOPIC, stream)

                if T_send is not None:

                    message = pose_message(
                        T_send,
                        request_id=gate.request_id,
                        object_name="tomato",
                        origin="center",
                        diameter=est.diameter,
                        settle_s=gate.settle_s,
                        # locked rotation = no yaw to grip by
                        symmetric=LOCK_ROTATION
                    )

                    link.publish(
                        POSE_TOPIC,
                        message
                    )

                    if gate.request_id is not None:
                        sent_poses[gate.request_id] = (message, time.monotonic())
                        if len(sent_poses) > 20:
                            sent_poses.pop(next(iter(sent_poses)))

                    print(
                        f"\nPose sent on {POSE_TOPIC} "
                        f"(request {gate.request_id}):\n{np.round(T_send, 4)}"
                    )

                    # Done until the next ready: the arm moves the camera now
                    is_registered = False
                    pose_matrix = None
                    lost_counter = 0
                    stable_counter = 0
                    previous_bbox = None

                else:

                    remaining = gate.remaining(
                        time.monotonic()
                    )

                    if remaining is not None:

                        cv2.putText(
                            display,
                            f"Holding still: {remaining:.1f} s to send",
                            (20, 138),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            2
                        )

                # =================================================
                # PRINT 6D POSE
                # =================================================

                if (
                    frame_count
                    % PRINT_EVERY
                    == 0
                ):

                    print_6d_pose(
                        pose_data
                    )

                # =================================================
                # POSE VISUALIZATION
                # =================================================

                display = draw_pose_axes(
                    display,
                    pose_data,
                    cam.K,
                    camera_dist,
                    axis_len=0.10
                )

                # =================================================
                # DISPLAY NUMERICAL POSE
                # =================================================

                # -------------------------------------------------
                # Position
                # -------------------------------------------------

                cv2.putText(
                    display,
                    (
                        f"X: {position[0]:.3f} m  "
                        f"Y: {position[1]:.3f} m  "
                        f"Z: {position[2]:.3f} m"
                    ),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2
                )

                # -------------------------------------------------
                # Euler
                # -------------------------------------------------

                cv2.putText(
                    display,
                    (
                        f"R: {euler[0]:.1f}  "
                        f"P: {euler[1]:.1f}  "
                        f"Y: {euler[2]:.1f} deg"
                    ),
                    (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2
                )

                # -------------------------------------------------
                # Quaternion
                # -------------------------------------------------

                cv2.putText(
                    display,
                    (
                        f"Q: "
                        f"{quat[0]:.2f},"
                        f"{quat[1]:.2f},"
                        f"{quat[2]:.2f},"
                        f"{quat[3]:.2f}"
                    ),
                    (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    2
                )

                # -------------------------------------------------
                # Distance
                # -------------------------------------------------

                cv2.putText(
                    display,
                    f"Distance: {distance:.3f} m",
                    (20, 112),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                # -------------------------------------------------
                # Tracking state
                # -------------------------------------------------

                cv2.putText(
                    display,
                    "FOUNDATIONPOSE 6D TRACKING",
                    (
                        20,
                        display.shape[0] - 45
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    "R = Reset | ESC = Exit",
                    (
                        20,
                        display.shape[0] - 18
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    2
                )

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                "FoundationPose Live 6D",
                display
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # =================================================
            # ESC
            # =================================================

            if key == 27:

                print(
                    "\nExiting..."
                )

                break

            # =================================================
            # R = RESET
            # =================================================

            # =================================================
            # SPACE = LOOK NOW (no ready message)
            # =================================================

            elif key == ord(" "):

                if gate.arm(None):

                    print(
                        "\nSPACE: looking for the tomato."
                    )

            elif key == ord("r"):

                print(
                    "\nManual reset."
                )

                gate.restart()

                is_registered = (
                    False
                )

                pose_matrix = (
                    None
                )

                lost_counter = (
                    0
                )

                stable_counter = (
                    0
                )

                previous_bbox = (
                    None
                )

                candidate_mask = (
                    None
                )

                candidate_bbox = (
                    None
                )

    finally:

        print(
            "\nStopping camera..."
        )

        cam.stop()

        cv2.destroyAllWindows()

        link.close()

        print(
            "Program terminated."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
