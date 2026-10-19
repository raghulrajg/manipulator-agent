"""
dobot_feedback.py
------------------
The controller's real-time feedback port (30004): one 1440-byte packet
every 8 ms with the robot's actual joint angles, mode, tool pose and queue
ID ("TCP_IP Remote Control Interface Guide V4.6.0", chapter 3, Real-time
Feedback). Read-only: any number of clients may listen while dashboard
commands go to port 29999, so the digital twin follows every motion of the
robot -- its own commands, the pendant, drag mode -- at full rate.

    reader = FeedbackReader("192.168.5.1").start()
    state = reader.latest()        # dict, see FIELDS; None until the first packet

pack_feedback() builds the same packet, for fake_dobot.py.

Run (read only):
    python3 dobot_feedback.py --ip 192.168.5.1
"""

import argparse
import socket
import struct
import threading
import time

import numpy as np

PACKET_SIZE = 1440
TEST_VALUE = 0x0123456789ABCDEF   # bytes 48-55 of every packet (memory layout check)
NOVA2_ROBOT_TYPE = 101

# name: (byte offset, struct format) -- little-endian, guide chapter 3
FIELDS = {
    "message_size": (0, "<H"),
    "robot_mode": (24, "<Q"),
    "timestamp_ms": (32, "<Q"),
    "test_value": (48, "<Q"),
    "speed_scaling": (64, "<d"),
    "q_target": (192, "<6d"),
    "q_actual": (432, "<6d"),
    "qd_actual": (480, "<6d"),
    "tool_vector_actual": (624, "<6d"),    # x y z [mm], rx ry rz [deg] of the selected tool (0 = flange)
    "enable_status": (1026, "<B"),
    "drag_status": (1027, "<B"),           # 0 no, 1 joint drag, 2 force-control drag
    "running_status": (1028, "<B"),
    "error_status": (1029, "<B"),
    "robot_type": (1031, "<B"),            # 101 = Nova 2
    "collision_state": (1038, "<B"),
    "current_command_id": (1112, "<Q"),
}
_VECTORS = ("q_target", "q_actual", "qd_actual", "tool_vector_actual")


class FeedbackError(ValueError):
    pass


def parse_feedback(packet):
    """One 1440-byte packet -> dict of FIELDS (vectors as numpy arrays)."""
    if len(packet) != PACKET_SIZE:
        raise FeedbackError(f"packet is {len(packet)} bytes, expected {PACKET_SIZE}")
    out = {}
    for name, (offset, fmt) in FIELDS.items():
        values = struct.unpack_from(fmt, packet, offset)
        out[name] = np.array(values) if name in _VECTORS else values[0]
    if out["test_value"] != TEST_VALUE:
        raise FeedbackError(f"test value {out['test_value']:#x}: not a feedback packet (out of step?)")
    if out["message_size"] != PACKET_SIZE:
        raise FeedbackError(f"message size {out['message_size']}, expected {PACKET_SIZE}")
    return out


def pack_feedback(**values):
    """Build a packet; fields not given are 0 (test_value and size are set)."""
    buf = bytearray(PACKET_SIZE)
    values = {"message_size": PACKET_SIZE, "test_value": TEST_VALUE, "robot_type": NOVA2_ROBOT_TYPE, **values}
    for name, value in values.items():
        offset, fmt = FIELDS[name]
        if name in _VECTORS:
            struct.pack_into(fmt, buf, offset, *[float(v) for v in value])
        else:
            struct.pack_into(fmt, buf, offset, value)
    return bytes(buf)


class FeedbackReader:
    """Background thread on port 30004 keeping the latest packet. Reconnects
    by itself; latest() is None until a packet arrives, and age() says how
    old the last one is (seconds, inf if none)."""

    def __init__(self, ip="192.168.5.1", port=30004, timeout=2.0, on_packet=None):
        self.ip, self.port, self.timeout = ip, port, timeout
        self.on_packet = on_packet
        self._state = None
        self._stamp = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.packets = 0
        self.resyncs = 0
        self.error = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="dobot-feedback")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def latest(self):
        with self._lock:
            return None if self._state is None else dict(self._state)

    def age(self):
        with self._lock:
            return float("inf") if self._stamp is None else time.monotonic() - self._stamp

    def wait_first(self, timeout=5.0):
        t0 = time.monotonic()
        while self.latest() is None:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(f"no feedback from {self.ip}:{self.port} after {timeout:.0f} s"
                                   + (f" ({self.error})" if self.error else ""))
            time.sleep(0.01)
        return self.latest()

    def _run(self):
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.ip, self.port), self.timeout) as sock:
                    sock.settimeout(self.timeout)
                    self.error = None
                    self._read(sock)
            except OSError as e:
                self.error = str(e)
            self._stop.wait(0.5)

    def _read(self, sock):
        buf = b""
        while not self._stop.is_set():
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("feedback connection closed")
            buf += chunk
            newest = None
            while len(buf) >= PACKET_SIZE:
                try:
                    newest = parse_feedback(buf[:PACKET_SIZE])
                    buf = buf[PACKET_SIZE:]
                except FeedbackError:
                    # out of step: find the next packet by its size + test value
                    self.resyncs += 1
                    buf = _resync(buf)
            if newest is not None:
                with self._lock:
                    self._state, self._stamp = newest, time.monotonic()
                self.packets += 1
                if self.on_packet:
                    self.on_packet(newest)


def _resync(buf):
    marker = struct.pack("<Q", TEST_VALUE)
    i = buf.find(marker, 49)
    while i != -1:
        start = i - 48
        if struct.unpack_from("<H", buf, start)[0] == PACKET_SIZE:
            return buf[start:]
        i = buf.find(marker, i + 1)
    return buf[-(PACKET_SIZE - 1):]   # keep a tail that may hold the start of one


def main():
    parser = argparse.ArgumentParser(description="Print the robot's real-time feedback (read only).")
    parser.add_argument("--ip", default="192.168.5.1")
    parser.add_argument("--port", type=int, default=30004)
    args = parser.parse_args()
    np.set_printoptions(precision=3, suppress=True)
    reader = FeedbackReader(args.ip, args.port).start()
    s = reader.wait_first()
    print(f"robot type {s['robot_type']} ({'Nova 2' if s['robot_type'] == NOVA2_ROBOT_TYPE else '?'})")
    try:
        while True:
            s = reader.latest()
            print(f"mode {s['robot_mode']:2d}  joints {s['q_actual']}  tool {s['tool_vector_actual'][:3]} mm  "
                  f"queue {s['current_command_id']}  drag {s['drag_status']}  {reader.packets} packets", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
