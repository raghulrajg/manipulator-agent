"""
camera_stream.py
-----------------
The wrist camera's live picture for the digital twin: the RealSense D405
(colour and depth, 640x480 at 30 fps, depth aligned to colour), read in a
thread and handed out as JPEG frames. digital_twin.py --camera serves
them to the browser, whose Camera view then shows the simulation, the
live picture, or the twin drawn over the live picture.

    RealSenseCamera("419122270335")   the D405 by serial number
    FakeCamera()                      a moving test picture (--sim, tests)

Only one program can open a RealSense at a time: if the vision PC's
FoundationPose script (or its router) holds the camera, run the twin
without --camera, or on the PC that has the camera free.

Run alone to check the camera (prints frames per second and intrinsics):
    python3 camera_stream.py --serial 419122270335
"""

import argparse
import math
import threading
import time

import numpy as np

D405_SERIAL = "419122270335"
WIDTH, HEIGHT, FPS = 640, 480, 30


class CameraError(RuntimeError):
    pass


class _Camera:
    """latest(kind) -> JPEG bytes of the newest frame ("color" or "depth"),
    info() -> state, intrinsics and error for the browser."""

    def __init__(self, name, jpeg_quality=80):
        self.name = name
        self.jpeg_quality = int(jpeg_quality)
        self.lock = threading.Lock()
        self._frames = {}          # kind -> (seq, image)
        self._jpeg = {}            # kind -> (seq, bytes)
        self.seq = 0
        self.stamp = None
        self.error = None
        self.intrinsics = None     # {"width", "height", "fx", "fy", "ppx", "ppy"}
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{self.name}")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _publish(self, color, depth_colormap=None, intrinsics=None):
        with self.lock:
            self.seq += 1
            self._frames["color"] = (self.seq, color)
            if depth_colormap is not None:
                self._frames["depth"] = (self.seq, depth_colormap)
            if intrinsics is not None:
                self.intrinsics = intrinsics
            self.stamp = time.time()
            self.error = None

    def latest(self, kind="color"):
        """JPEG bytes of the newest frame, encoded once per frame however
        many browsers watch; None until there is one."""
        import cv2
        with self.lock:
            frame = self._frames.get(kind)
            cached = self._jpeg.get(kind)
        if frame is None:
            return None, None
        seq, image = frame
        if cached is not None and cached[0] == seq:
            return seq, cached[1]
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return None, None
        data = buf.tobytes()
        with self.lock:
            self._jpeg[kind] = (seq, data)
        return seq, data

    def info(self):
        with self.lock:
            age = None if self.stamp is None else time.time() - self.stamp
            return {"name": self.name, "live": age is not None and age < 2.0, "age_s": None if age is None
                    else round(age, 2), "error": self.error, "intrinsics": self.intrinsics,
                    "kinds": sorted(self._frames)}

    def _run(self):
        raise NotImplementedError


class RealSenseCamera(_Camera):
    """The D405 by serial number, as in your test script: colour + depth,
    depth aligned to colour, depth shown with the JET colormap. Reconnects
    every 2 s if the camera is unplugged or held by another program."""

    def __init__(self, serial=D405_SERIAL, width=WIDTH, height=HEIGHT, fps=FPS, **kw):
        super().__init__(f"RealSense {serial}", **kw)
        self.serial, self.width, self.height, self.fps = str(serial), int(width), int(height), int(fps)

    def _run(self):
        try:
            import cv2
            import pyrealsense2 as rs
        except ImportError as e:
            self.error = f"{e.name} is not installed (pip install pyrealsense2 opencv-python)"
            return
        while not self._stop.is_set():
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            try:
                profile = pipeline.start(config)
            except RuntimeError as e:
                self.error = f"cannot open camera {self.serial}: {e} (unplugged, or another program has it)"
                self._stop.wait(2.0)
                continue
            try:
                k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                intrinsics = {"width": k.width, "height": k.height, "fx": k.fx, "fy": k.fy,
                              "ppx": k.ppx, "ppy": k.ppy}
                align = rs.align(rs.stream.color)
                while not self._stop.is_set():
                    frames = align.process(pipeline.wait_for_frames(5000))
                    depth, color = frames.get_depth_frame(), frames.get_color_frame()
                    if not depth or not color:
                        continue
                    color_image = np.asanyarray(color.get_data()).copy()
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(np.asanyarray(depth.get_data()), alpha=0.03), cv2.COLORMAP_JET)
                    self._publish(color_image, depth_colormap, intrinsics)
            except RuntimeError as e:
                self.error = f"camera {self.serial}: {e}"
            finally:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
            self._stop.wait(2.0)


class FakeCamera(_Camera):
    """A moving test picture with the D405's intrinsics, for --sim and the
    tests: no camera needed."""

    def __init__(self, fps=15, **kw):
        super().__init__("simulated camera", **kw)
        self.fps = fps

    def _run(self):
        import cv2
        intrinsics = {"width": WIDTH, "height": HEIGHT, "fx": 433.0, "fy": 433.0, "ppx": WIDTH / 2, "ppy": HEIGHT / 2}
        t0 = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic() - t0
            img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            img[:] = (60, 70, 80)
            cx = int(WIDTH / 2 + 120 * math.sin(t))
            cv2.circle(img, (cx, HEIGHT // 2), 40, (40, 60, 220), -1)
            cv2.putText(img, f"simulated camera {t:6.1f} s", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (230, 230, 230), 2)
            depth = cv2.applyColorMap(np.full((HEIGHT, WIDTH), int(128 + 100 * math.sin(t)), np.uint8),
                                      cv2.COLORMAP_JET)
            self._publish(img, depth, intrinsics)
            self._stop.wait(1.0 / self.fps)


def open_camera(spec):
    """--camera value -> a started camera: "fake" or a RealSense serial."""
    cam = FakeCamera() if spec == "fake" else RealSenseCamera(spec)
    return cam.start()


def main():
    p = argparse.ArgumentParser(description="Check the wrist camera: frames per second and intrinsics.")
    p.add_argument("--serial", default=D405_SERIAL, help=f"RealSense serial (default {D405_SERIAL}), or 'fake'")
    p.add_argument("--seconds", type=float, default=5.0)
    args = p.parse_args()
    cam = open_camera(args.serial)
    t0, first = time.monotonic(), None
    try:
        while time.monotonic() - t0 < args.seconds:
            time.sleep(0.5)
            info = cam.info()
            if info["error"]:
                print(info["error"])
            elif info["live"] and first is None:
                first = (time.monotonic(), cam.seq)
        if first is None:
            raise SystemExit(f"no frames from {cam.name}")
        fps = (cam.seq - first[1]) / max(time.monotonic() - first[0], 1e-6)
        print(f"{cam.name}: {fps:.1f} frames/s, colour intrinsics {cam.info()['intrinsics']}")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
