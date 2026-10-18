"""
pick_runner.py
---------------
The pick on the real arm: MQTT to the vision PC (foundationpose_mqtt.py),
TCP to the Dobot controller (dobot_tcp.py), RS485 to the Feetech gripper
(gripper.py). The controller plans every motion itself; this only sends the
joint targets the IK chose and checks each one is reached before going on.

One cycle:
  1. arm at the detection pose (home), still; gripper open
  2. publish "ready" -> the vision PC finds the object, waits until it has
     held still for 2.5 s and publishes T_camera_object
  3. T_base_object = FK_camera(joints read now) @ T_camera_object, then the
     table-safe plan (manipulator_agent): pre-grasp, the straight line down,
     the lift, the place and the way home -- all checked before anything moves
  4. MovJ to the pre-grasp, MovL straight down to the grasp
  5. measure before gripping: the joints (GetAngle -> FK) and the controller's
     own flange position (GetPose) must both be at the grasp
  6. close the gripper; it has to stop on the object, not close all the way
  7. MovL straight up; with --place: MovJ above the place spot, MovL down,
     open, MovL up
  8. MovJ home, publish the status, next cycle

A missed grip opens the gripper, goes back up and home, and reports
"failed". Any robot alarm, collision, lost connection or gripper fault stops
the robot and the script.

Table height: --table-z, the table surface in the base frame, measured
once (touch it with the tool tip and read the pose). It is required on the
real robot; only --dry-run may estimate it from the message (the object's
center minus half its diameter).

Gripper: --arm left / right, by default paired with the robot IP the way
x-trainer does (192.168.5.1 left = ID 21 on /dev/ttyUSB1, 192.168.5.2
right = ID 22 on /dev/ttyUSB0).

Run:
    python3 pick_runner.py --table-z -0.05                       # pick once, hold it, return home
    python3 pick_runner.py --table-z -0.05 --place -0.25 0.40 --cycles 0     # pick and place, forever
    python3 pick_runner.py --dry-run --fake-vision demo          # no robot, gripper, camera or broker
"""

import argparse
import math
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from dobot_tcp import MODE_IDLE, DobotError, DobotTCP
from gripper import LEFT, RIGHT, FakeGripper, FeetechGripper, GripperError
from manipulator_agent import ManipulatorAgent, plan_place
from manipulator_kinematics import IKError, T_FLANGE_TOOL, fk, joint_map, pose
from mqtt_link import (
    POSE_TOPIC, READY_TOPIC, STATUS_TOPIC, LocalBus, MessageError, MqttLink,
    parse_pose_message, parse_ready, pose_message, ready_message,
)

READY_REPEAT_S = 5.0   # send "ready" again this often until a pose arrives
ROT_TOL_DEG = 1.0      # tool orientation tolerance at a target


class PickFailed(RuntimeError):
    """This pick did not work, but the arm is back home and safe."""


@dataclass
class Move:
    name: str
    kind: str                    # "movj" or "movl"
    q: np.ndarray                # joint target, servo deg
    T_tool: np.ndarray           # tool tip pose there
    then: Optional[str] = None   # "close" / "open" the gripper once there


def tool_error(servo_deg, T_target):
    """(position error [m], rotation error [deg]) of the tool tip at these
    joints against T_target."""
    T = fk(joint_map.servo_to_visual(servo_deg), "tool")
    R = T[:3, :3].T @ T_target[:3, :3]
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    return float(np.linalg.norm(T[:3, 3] - T_target[:3, 3])), angle


def _yaw(T):
    """Heading of a frame's x axis about the base z [rad]."""
    return math.atan2(T[1, 0], T[0, 0])


def table_from_message(msg, T_base_object):
    """Table height from the object: it rests on the table."""
    origin = msg.get("origin", "center")
    if origin == "bottom":
        return float(T_base_object[2, 3])
    if origin == "center" and msg.get("diameter"):
        return float(T_base_object[2, 3]) - float(msg["diameter"]) / 2
    raise IKError(f"cannot tell the table height from the message (origin {origin!r}, "
                  "no diameter): pass --table-z")


def plan_moves(agent, T_camera_object, q_detect, home, place_xy=None):
    """Every move of one pick (and place), checked against the table before
    the arm moves. Returns (PickPlan, [Move])."""
    pick = agent.plan_pick(T_camera_object, q_detect, q_detect)
    table = agent.table_for(pick.T_base_object)
    wp = pick.waypoints
    pre = wp[0]
    grasp = next(w for w in wp if w.name == "grasp")
    moves = [Move("pre_grasp", "movj", pre.q_servo_deg, pre.T_base_tool),
             Move("grasp", "movl", grasp.q_servo_deg, grasp.T_base_tool, then="close"),
             Move("lift", "movl", wp[-1].q_servo_deg, wp[-1].T_base_tool)]
    q_last = wp[-1].q_servo_deg

    if place_xy is not None:
        # the object's origin goes to place_xy at the height it was picked from
        T_place_tool = grasp.T_base_tool.copy()
        T_place_tool[:2, 3] += np.asarray(place_xy, dtype=float) - pick.T_base_object[:2, 3]
        try:
            line = plan_place(agent, table, T_place_tool, q_last)
        except IKError as e:
            raise IKError(f"place spot {np.round(place_xy, 4).tolist()}: {e}") from None
        (_, T_pre, q_pre), (_, T_put, q_put) = line[0], line[-1]
        moves += [Move("pre_place", "movj", q_pre, T_pre),
                  Move("place", "movl", q_put, T_put, then="open"),
                  Move("retreat", "movl", q_pre, T_pre)]
        q_last = q_pre

    bad = table.path_violations(joint_map.servo_to_visual(q_last), joint_map.servo_to_visual(home))
    if bad:
        raise IKError("the way back home goes below the table: " + ", ".join(bad))
    moves.append(Move("home", "movj", np.asarray(home, float), fk(joint_map.servo_to_visual(home), "tool")))
    return pick, moves


class PickRunner:
    def __init__(self, robot, gripper, link, home=None, table_z=None, place_xy=None,
                 agent_options=None, speed=20, line_speed=10, speed_factor=None, joint_tol_deg=0.3,
                 reach_tol_mm=2.0, model_tol_mm=3.0, pose_timeout=None, log=print):
        self.robot, self.gripper, self.link = robot, gripper, link
        self.home = np.asarray(joint_map.home_servo_deg if home is None else home, dtype=float)
        self.table_z, self.place_xy = table_z, place_xy
        self.agent_options = dict(agent_options or {})
        self.speed, self.line_speed, self.speed_factor = speed, line_speed, speed_factor
        self.joint_tol_deg, self.reach_tol_mm, self.model_tol_mm = joint_tol_deg, reach_tol_mm, model_tol_mm
        self.pose_timeout = pose_timeout
        self.log = log
        self.request_id = None
        self.pose_queue = link.subscribe(POSE_TOPIC)

    def status(self, state, detail=""):
        self.log(f"[{state}] {detail}".rstrip())
        self.link.publish(STATUS_TOPIC, {"request_id": self.request_id, "state": state,
                                         "detail": detail, "stamp": time.time()})

    # ------------------------------------------------------------ robot checks
    def check_model(self):
        """The controller's flange position (GetPose, tool 0) must match the
        FK of its joints, or the IK would send the arm somewhere else."""
        q = self.robot.get_angle()
        measured = self.robot.get_pose()[:3] / 1000
        model = fk(joint_map.servo_to_visual(q), "flange")[:3, 3]
        err = np.linalg.norm(measured - model) * 1000
        self.log(f"model check: GetPose {np.round(measured * 1000, 1).tolist()} mm, "
                 f"FK {np.round(model * 1000, 1).tolist()} mm -> {err:.2f} mm")
        if err > self.model_tol_mm:
            raise DobotError(f"the kinematic model is {err:.1f} mm off the controller's own pose: "
                             "see 'Correction to your DH table' in the README before running")

    def at_target(self, move):
        """Measure before using the gripper: the joints (FK) and the
        controller's flange position (GetPose) must both be at the target."""
        q = self.robot.get_angle()
        dp, dang = tool_error(q, move.T_tool)
        flange = (move.T_tool @ np.linalg.inv(T_FLANGE_TOOL))[:3, 3]
        dctrl = np.linalg.norm(self.robot.get_pose()[:3] / 1000 - flange) * 1000
        self.log(f"at {move.name}: tool tip {dp * 1000:.2f} mm / {dang:.2f} deg from the target, "
                 f"controller flange {dctrl:.2f} mm")
        if dp * 1000 > self.reach_tol_mm or dang > ROT_TOL_DEG or dctrl > self.reach_tol_mm + self.model_tol_mm:
            raise DobotError(f"not at {move.name}: tool tip {dp * 1000:.1f} mm / {dang:.1f} deg off, "
                             f"controller flange {dctrl:.1f} mm off")

    def _reached(self, T_target):
        def reached(q):
            dp, dang = tool_error(q, T_target)
            return dp * 1000 <= self.reach_tol_mm and dang <= ROT_TOL_DEG
        return reached

    def move(self, m):
        if m.kind == "movj":
            return self.robot.move_j(m.q, v=self.speed, a=self.speed, tol_deg=self.joint_tol_deg, what=m.name)
        return self.robot.move_l(m.q, self._reached(m.T_tool), v=self.line_speed, a=self.line_speed, what=m.name)

    # ------------------------------------------------------------ cycle
    def startup(self):
        try:
            self.robot.request_control()
        except DobotError as e:
            self.log(f"RequestControl: {e} (already in TCP mode?)")
        self.robot.enable()
        if self.speed_factor is not None:
            self.robot.speed_factor(self.speed_factor)
        self.check_model()
        self.gripper.open()
        q = self.robot.get_angle()
        if np.abs(q - self.home).max() > self.joint_tol_deg:
            if self.table_z is not None:
                table = ManipulatorAgent(table_z=self.table_z, **self._agent_kwargs()).table_for(np.eye(4))
                bad = table.path_violations(joint_map.servo_to_visual(q), joint_map.servo_to_visual(self.home))
                if bad:
                    raise IKError("the move home from here goes below the table (" + ", ".join(bad)
                                  + "): jog the arm closer to home first")
            else:
                self.log("warning: the first move home is not checked against the table (no --table-z)")
            self.move(Move("home", "movj", self.home, fk(joint_map.servo_to_visual(self.home), "tool")))

    def _agent_kwargs(self):
        return {k: v for k, v in self.agent_options.items() if v is not None}

    def wait_pose(self, joints):
        """Publish ready and wait for the pose answering it."""
        t0 = sent = time.monotonic()
        self.link.publish(READY_TOPIC, ready_message(self.request_id, joints))
        self.status("waiting", f"ready sent on {READY_TOPIC}, waiting for {POSE_TOPIC}")
        while True:
            try:
                payload = self.pose_queue.get(timeout=0.2)
            except queue.Empty:
                now = time.monotonic()
                if self.pose_timeout and now - t0 > self.pose_timeout:
                    raise PickFailed(f"no object pose after {self.pose_timeout:.0f} s")
                if now - sent > READY_REPEAT_S:
                    self.link.publish(READY_TOPIC, ready_message(self.request_id, joints))
                    sent = now
                continue
            try:
                msg = parse_pose_message(payload)
            except MessageError as e:
                self.log(f"ignoring pose message: {e}")
                continue
            if msg["request_id"] not in (None, self.request_id):
                self.log(f"ignoring a pose for request {msg['request_id']} (waiting for {self.request_id})")
                continue
            return msg

    def cycle(self):
        """One detect -> pick (-> place) -> home. Returns the PickPlan."""
        self.request_id = uuid.uuid4().hex[:8]
        self.gripper.open()
        while not self.pose_queue.empty():
            self.pose_queue.get_nowait()   # nothing old
        if self.robot.robot_mode() != MODE_IDLE:
            raise DobotError("robot is not enabled and idle")
        q_detect = self.robot.get_angle()
        msg = self.wait_pose(q_detect)
        if np.abs(self.robot.get_angle() - q_detect).max() > 0.05:
            raise DobotError("the arm moved while the camera was looking")

        T_camera_object = msg["T_camera_object"]
        T_base_object = fk(joint_map.servo_to_visual(q_detect), "camera") @ T_camera_object
        table_z = self.table_z if self.table_z is not None else table_from_message(msg, T_base_object)
        options = self._agent_kwargs()
        options.setdefault("grasp_height", 0.0 if msg.get("origin") == "center" else 0.02)
        if msg.get("symmetric"):
            # round object, no rotation to go by: keep the gripper turned as it is now
            options["grasp_yaw_offset_deg"] = options.get("grasp_yaw_offset_deg", 0.0) + math.degrees(
                _yaw(fk(joint_map.servo_to_visual(q_detect), "tool")) - _yaw(T_base_object))
        agent = ManipulatorAgent(table_z=table_z, **options)
        plan, moves = plan_moves(agent, T_camera_object, q_detect, self.home, self.place_xy)
        self.log(f"object {msg.get('object', '')} at {np.round(T_base_object[:3, 3], 4).tolist()} m "
                 f"(base frame), table z {table_z:.4f} m")
        for m in moves:
            self.log(f"  {m.name:<9} {m.kind}  servo deg {np.round(m.q, 2).tolist()}  "
                     f"tool tip {np.round(m.T_tool[:3, 3], 4).tolist()}" + (f"  then {m.then}" if m.then else ""))

        self.status("picking", f"object at {np.round(T_base_object[:3, 3], 4).tolist()}")
        by_name = {m.name: m for m in moves}
        for m in moves:
            self.move(m)
            if m.then == "close":
                self.at_target(m)
                grip = self.gripper.close()
                self.log(f"gripper: position {grip.position}/255, load {grip.load}")
                if not grip.holding:
                    self.gripper.open()
                    self.move(by_name["lift"])
                    self.move(by_name["home"])
                    raise PickFailed("the gripper closed on nothing")
                self.status("holding")
            elif m.then == "open":
                self.at_target(m)
                self.gripper.open()
                self.status("placed", f"at {np.round(self.place_xy, 4).tolist()}")
        self.status("done", "holding the object at home" if self.place_xy is None else "")
        return plan


def start_fake_vision(bus, T_camera_object, diameter=0.06, delay=0.2):
    """Answer every ready on the bus with this pose (for --fake-vision)."""
    ready_q = bus.subscribe(READY_TOPIC)

    def run():
        answered = set()
        while True:
            ready = parse_ready(ready_q.get())
            if ready["request_id"] in answered:
                continue
            answered.add(ready["request_id"])
            time.sleep(delay)
            bus.publish(POSE_TOPIC, pose_message(T_camera_object, ready["request_id"], "demo", "center",
                                                 diameter, symmetric=True))

    threading.Thread(target=run, daemon=True).start()


DEMO_T_CAMERA_OBJECT = pose(np.eye(3), [0.03, -0.02, 0.38])   # 6 cm tomato, 38 cm below the camera


def main():
    p = argparse.ArgumentParser(description="Pick the object the vision PC finds (MQTT -> IK -> Dobot -> gripper).")
    p.add_argument("--robot-ip", default=LEFT["robot_ip"])
    p.add_argument("--arm", choices=("left", "right"),
                   help="which gripper: left = ID 21 on /dev/ttyUSB1, right = ID 22 on /dev/ttyUSB0 "
                   "(default: as x-trainer pairs them, 192.168.5.1 left, 192.168.5.2 right)")
    p.add_argument("--gripper-port", help="override the arm's gripper port")
    p.add_argument("--gripper-id", type=int, help="override the arm's gripper servo ID")
    p.add_argument("--grip-torque", type=int, default=300, help="gripper torque limit 0..1000 (default 300)")
    p.add_argument("--table-z", type=float, help="table surface height in the base frame [m] "
                   "(required on the real robot)")
    p.add_argument("--place", type=float, nargs=2, metavar=("X", "Y"),
                   help="put the object down here (base frame, m) after picking")
    p.add_argument("--cycles", type=int, default=1, help="picks to do, 0 = forever (needs --place)")
    p.add_argument("--home", type=float, nargs=6, metavar="J", help="detection pose, servo deg (default home)")
    p.add_argument("--grasp-height", type=float, help="tool tip above the object's origin at the grasp [m] "
                   "(default 0 for a center origin, 0.02 for a bottom origin)")
    p.add_argument("--approach", type=float, default=0.10, help="pre-grasp height above the grasp [m]")
    p.add_argument("--grasp-yaw", type=float, default=0.0, help="gripper turn about the vertical [deg]")
    p.add_argument("--speed", type=int, default=20, help="MovJ speed and acceleration %% (1..100)")
    p.add_argument("--line-speed", type=int, default=10, help="MovL speed and acceleration %% (1..100)")
    p.add_argument("--speed-factor", type=int, help="set the controller's global speed %% (SpeedFactor) at "
                   "start; default: leave it as set on the pendant")
    p.add_argument("--reach-tol-mm", type=float, default=2.0, help="how close the tool tip must be [mm]")
    p.add_argument("--pose-timeout", type=float, help="give up waiting for the vision after this many s")
    p.add_argument("--dry-run", action="store_true", help="fake robot and gripper (fake_dobot.py)")
    p.add_argument("--fake-vision", metavar="POSE", help="answer ready ourselves with this 4x4 T_camera_object "
                   "file, or 'demo' (with --dry-run only); no broker needed")
    args = p.parse_args()
    if args.place is None and args.cycles != 1:
        p.error("without --place the gripper is still holding the object after a pick: use --cycles 1")
    if not args.dry_run:
        if args.table_z is None:
            p.error("--table-z is required on the real robot: the table height must not come from the "
                    "same camera pose the grasp does")
        if args.fake_vision == "demo":
            p.error("--fake-vision demo is a made-up object: use it with --dry-run")
    home = joint_map.home_servo_deg if args.home is None else np.asarray(args.home, float)
    home_tip_z = fk(joint_map.servo_to_visual(home), "tool")[2, 3]
    if args.table_z is not None and not -1.0 < args.table_z < home_tip_z - 0.05:
        p.error(f"--table-z {args.table_z} m is not between -1 m and 5 cm below the tool tip at home "
                f"({home_tip_z:.3f} m): it is in meters")
    for name in ("speed", "line_speed", "speed_factor"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= 100:
            p.error(f"--{name.replace('_', '-')} must be 1..100")
    if args.arm is None:
        sides = {LEFT["robot_ip"]: "left", RIGHT["robot_ip"]: "right"}
        if args.robot_ip not in sides and not args.dry_run:
            p.error(f"which gripper goes with {args.robot_ip}? pass --arm left or --arm right")
        args.arm = sides.get(args.robot_ip, "left")
    np.set_printoptions(precision=4, suppress=True)

    if args.fake_vision:
        from pick_from_pose import load_pose
        link = LocalBus()
        start_fake_vision(link, DEMO_T_CAMERA_OBJECT if args.fake_vision == "demo" else load_pose(args.fake_vision))
    else:
        link = MqttLink(client_id=f"pick_runner_{uuid.uuid4().hex[:6]}")

    fake = None
    if args.dry_run:
        from fake_dobot import FakeDobot
        fake = FakeDobot().start()
        robot = DobotTCP("127.0.0.1", fake.port)
        gripper = FakeGripper()
    else:
        robot = DobotTCP(args.robot_ip)
        side = LEFT if args.arm == "left" else RIGHT
        gripper = FeetechGripper(args.gripper_port or side["port"], args.gripper_id or side["servo_id"],
                                 torque_limit=args.grip_torque)
        print(f"robot {args.robot_ip}, {args.arm} gripper: servo ID {gripper.servo_id} on {gripper.port}")

    runner = PickRunner(robot, gripper, link, home, args.table_z, args.place,
                        dict(approach_height=args.approach, grasp_yaw_offset_deg=args.grasp_yaw,
                             grasp_height=args.grasp_height),
                        speed=args.speed, line_speed=args.line_speed, speed_factor=args.speed_factor,
                        reach_tol_mm=args.reach_tol_mm, pose_timeout=args.pose_timeout)
    stop_robot, code, n, failures = True, 1, 0, 0   # anything unexpected stops the robot
    try:
        robot.connect()
        link.connect()
        runner.startup()
        while args.cycles == 0 or n < args.cycles:
            n += 1
            try:
                runner.cycle()
            except (PickFailed, IKError) as e:   # the arm is home (or never left)
                runner.status("failed", str(e))
                failures += 1
                time.sleep(1.0)
        stop_robot, code = False, 1 if failures else 0
    except KeyboardInterrupt:
        print("\nstopping")
        code = 130
    except (DobotError, GripperError, IKError, ConnectionError, OSError) as e:
        print(f"error: {e}")
    finally:
        if stop_robot:
            try:
                robot.stop()
            except (DobotError, OSError):
                pass
        try:
            runner.status("stopped" if stop_robot else "finished", f"{n} cycles, {failures} failed")
        except Exception:
            pass
        robot.close()
        gripper.disconnect()
        link.close()
        if fake:
            fake.stop()
    sys.exit(code)


if __name__ == "__main__":
    main()
