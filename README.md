# Manipulator control: object pose and table-safe tool-tip IK

Pick planning for the 6-DOF arm, built on your three DH tables: the flange
with no end effector, the gripper tool tip, and the wrist camera, and the
real-time pick on the Nova 2: MQTT to the vision PC, TCP/IP to the Dobot
controller, RS485 to the Feetech gripper (same hardware as x-trainer). It is
plain Python with no ROS2: the kinematics and planning need only numpy, the
robot side adds paho-mqtt and pyserial, the simulation matplotlib, the
digital twin aiohttp (and a browser).

| File | What it does |
|---|---|
| `manipulator_kinematics.py` | Your DH tables (three offsets corrected, see below) and `JointMap` (servo ↔ model angles), FK for flange / tool / camera, closed-form IK (up to 8 solutions), joint limits, and the table check. |
| `manipulator_agent.py` | `T_base_object` from a camera detection, then a top-down pick plan (pre-grasp → descend → grasp → lift) whose every waypoint and move stays above the table. Also a command-line demo. |
| `simulate_pick.py` | 3D simulation (matplotlib) of the arm picking the object and placing it elsewhere, using the same IK and planner as the robot. Before drawing, it checks every move at 1° steps: the arm and the carried object stay above the table, and the object is put down exactly at the place spot (position and orientation). Saves a GIF or MP4, or opens a window. |
| `pick_from_pose.py` | Full 6-DoF object pose (FoundationPose 4x4, position + orientation) → approach and grasp joint waypoints along the object's own axes; checked against the table (waypoints and the moves between them) when you pass `--table-z`. |
| `fk_from_joints.py` | Joint angles (servo deg, given or random) → flange, tool tip and camera poses; `--steps` prints every joint transform. |
| `ik_from_pose.py` | Pose of the tool tip, flange or camera (4x4 or x y z), in the base frame or relative to an object → tool tip pose in the base frame and joint angles, each checked with FK. See "Pose → joint angles" below. |
| `pick_runner.py` | **The real-time pick.** Sends "ready" over MQTT, takes the pose the vision PC sends back, plans with the IK and table check, drives the arm with Dobot MovJ / MovL, checks the target is reached (joints and the controller's own pose) before gripping, grips, lifts, places, goes home. See "Real-time pick on the robot" below. |
| `dobot_tcp.py` | Nova 2 TCP/IP client (controller V4 on port 29999 as in the guide in `Docs/`, or V3 with motion on port 30003, detected automatically): enable, MovJ / MovL to joint targets, wait until done (queue ID, RobotMode, reached), GetAngle, GetPose, alarms. Run it alone to read the robot's state. |
| `gripper.py` | Feetech SMS/STS gripper through `scservo_sdk/` (your SDK, unchanged), with x-trainer's settings: left arm (robot 192.168.5.1) ID 21 on /dev/ttyUSB1, right arm (192.168.5.2) ID 22 on /dev/ttyUSB0, raw 3052 open … 2048 closed (x-trainer's file has it the other way round; on this gripper 2048 is closed, `--open-raw` / `--closed-raw` change it). `close()` tells whether the fingers stopped on something. |
| `scservo_sdk/` | Feetech's SCServo Python SDK (SMS/STS and SCSCL servos, port handler), your copy unchanged. `gripper.py` uses it; it needs pyserial. |
| `mqtt_link.py` | Shared by both sides: topics, the ready / pose / status messages, the vision agent's commands (start / stop / idle, pick / place, position + quaternion), quaternion conversions, the paho-mqtt client (EMQX: username, TLS, WebSocket), and `PoseGate` (ready → found → still 2.5 s → send once). |
| `mqtt_send.py` | Send a command to the manipulator agent by hand, as the vision agent does, and print its replies. |
| `foundationpose_mqtt.py` | Your FoundationPose tomato script with MQTT added: idle until "ready", then detect, track, wait until the pose has held 2.5 s, publish it once, idle again. Runs on the vision PC. |
| `digital_twin.py` + `twin_web/` | **Digital twin**: a browser app with Dobot's own Nova 2 model that moves exactly as the robot does (real-time feedback) and moves the robot from the browser (checked MovJ / MovL, gripper, pick and place). `--sim` runs it on the fake controller. See "Digital twin" below. |
| `dobot_feedback.py` | The controller's real-time feedback port (30004): the 1440-byte packet every 8 ms (actual joints, mode, tool pose, queue ID, drag state), parsed at the guide's byte offsets, read in a background thread. |
| `fake_dobot.py` | A fake Nova 2 controller on localhost: dashboard commands and replies, a real-time feedback port, Nova 2 speeds (135°/s), straight-line MovL, drag mode, and moves "from the pendant". Used by `--dry-run`, `--sim` and the tests. |
| `test_manipulator.py` | Offline checks: FK against Dobot's official Nova 2 model and the DH tables, IK round trip, table constraint, full detect → pick plan. |
| `test_ik_random.py` | Random joint angles → FK (tool tip transform) → IK → joint angles, compared with the originals. Also covers joints exactly at their limits and every singular pose. |
| `test_pick_runner.py` | Offline checks of the robot side: the Dobot client against the fake controller (collision, stopping short, old firmware, limits), the gripper against a fake servo, the MQTT messages and `PoseGate`, and whole pick cycles (success, missed grip, unreachable, arm moved, wrong model, collision). `--broker HOST` also goes through a real MQTT broker. |
| `test_digital_twin.py` | Offline checks of the twin: the feedback packet layout, the URDF model against the kinematics, both directions of mirroring (incl. pendant and drag-mode moves), refusals (disarmed, limits, table, busy), STOP, pick and place, Detect, the WebSocket protocol, and a real browser run (with Playwright). |
| `test_vision_mqtt.py` | Runs `foundationpose_mqtt.py`'s real main loop with fake camera / YOLO / SAM / FoundationPose / broker: nothing before a ready; a pose only after 2.5 s of steady tracking, with the ready's request_id; a repeat of an answered ready gets the same pose again within 10 s and a fresh detection after that; losing the tomato, or holding its last pose while depth drops out, restarts the wait. |
| `Docs/` | Dobot's TCP/IP Remote Control Interface Guide V4.6.0 (the commands `dobot_tcp.py` sends) and the Nova series user guide (PDF). |
| `Helpers scripts/` | Your original scripts (`joint_mapping.py`, `effective_transformation.py`, `tooltip_transformation.py`, `camera_tranformation.py`, `nova2_control.py`), kept for reference. They still have the old J1/J3/J5 offsets (see the correction section). |
| `requirements.txt` | numpy, plus matplotlib (simulation), paho-mqtt (MQTT) and pyserial (gripper). |

```
pip install -r requirements.txt
python3 pick_from_pose.py --pose pose.txt --table-z -0.05   # FoundationPose 4x4 -> joint angles
python3 fk_from_joints.py --joints -180 0 -90 0 90 90   # joint angles -> flange / tool / camera poses
python3 ik_from_pose.py                          # home pose of tool / flange / camera + random points
python3 ik_from_pose.py --frame camera --matrix T_base_camera.txt
python3 ik_from_pose.py --xyz -0.40 0.10 0.05 --yaw 30 --table-z -0.05
python3 ik_from_pose.py --object T_camera_object.txt --frame camera --xyz 0 0 0.30
python3 simulate_pick.py                         # pick-and-place simulation -> pick.gif
python3 simulate_pick.py --object -0.45 0.05 0.0 30 --place -0.25 0.40 --save pick.mp4
python3 simulate_pick.py --pose T_camera_object.txt --show   # 4x4 object pose (bottom-centre origin, +z up), live window
python3 test_ik_random.py                        # 1000 random poses, first 3 printed in full
python3 test_ik_random.py --n 10000 --seed 42 --show 5
python3 test_manipulator.py
python3 manipulator_agent.py 0.02 -0.03 0.30 20      # object x y z [m] and yaw [deg] in the camera frame
python3 manipulator_agent.py 0.02 -0.03 0.30 20 --table-z -0.05
python3 test_pick_runner.py                      # robot side, no hardware (add --broker localhost)
python3 test_vision_mqtt.py                      # vision MQTT flow, no camera or GPU
python3 pick_runner.py --dry-run --fake-vision demo   # whole pick cycle, no hardware or broker
python3 digital_twin.py --sim                    # digital twin on a simulated robot (open the address it prints)
python3 test_digital_twin.py                     # digital twin, no hardware
```

`simulate_pick.py --pose` needs the object frame at its bottom centre with
+z up. FoundationPose's tomato pose has its origin at the centre and its
axes parallel to the camera, so convert it first, or the script refuses it.

## Digital twin (`digital_twin.py`)

A browser app that shows the Nova 2 as Dobot's own model and moves it. The
model is the URDF and meshes from
[Dobot-Arm/TCP-IP-ROS-6AXis](https://github.com/Dobot-Arm/TCP-IP-ROS-6AXis),
MIT-licensed and kept in `twin_web/model/nova2`. The twin follows the robot,
and the robot follows the twin:

```
python3 digital_twin.py --sim                                        # simulated robot; open the address it prints
python3 digital_twin.py --robot 192.168.5.1 --table-z 0.018          # the real arm and its gripper
python3 digital_twin.py --robot 192.168.5.1 --table-z 0.018 --mqtt   # + Detect with the vision PC
python3 digital_twin.py --sim --fake-vision demo                     # Detect answered with a demo pose
python3 digital_twin.py --sim --mqtt                                 # simulated robot, commands from your EMQX
python3 mqtt_send.py idle --position 0.02 -0.03 0.30 --orientation 0 0 0.383 0.924   # the object at a pose (see below)
python3 dobot_feedback.py --ip 192.168.5.1                           # just print the feedback (read only)
```

**Robot → twin.** The controller's real-time feedback port (30004) sends
the actual joint angles every 8 ms. The server pushes them to every open
browser 30 times a second. The twin therefore shows whatever moves the arm:
the app, the pendant, `pick_runner.py`, or a hand in drag mode. The app's
own moves reach the screen the same way, so the twin never shows a motion
the robot didn't make.

**Twin → robot.** A browser command is checked on the server and sent to
the robot as MovJ / MovL on port 29999 (the gripper over RS485). You then
see it move through the feedback.

| Panel | What it does |
|---|---|
| Robot | Enable, Disable, Clear alarm. Drag mode lets you move the arm by hand while the twin follows, and disarms the app. Speed is capped by `--max-speed`, default 50 %. |
| Joints | Sliders preview a target as a see-through ghost, checked against the limits and the table. **Move joints** sends it, ± jogs one joint. |
| Tool tip | Jog ±X / ±Y / ±Z as straight lines (MovL, the line checked every 5 mm). **Target** puts a tool target in the 3D view that you drag (T to move, R to turn); it previews the IK and then does **Move (joint)** or **Move (line)**. **Home**. |
| Gripper | Open, Close. |
| Pick & place | **Set object** / **Set place**, then click on the table; Ø sets the size. **Plan** shows the path and the grasp as a ghost. **Pick & place** runs `pick_runner.py`'s pick, including the reach check before gripping, and the tomato rides in the fingers. **Detect** (with `--mqtt`) goes home, sends ready and shows the object the vision PC finds. |
| View | 3D, Top, Front, and **Camera**, which looks through the wrist camera. |

**Access.** The server prints the address to open, including an access
token: `http://localhost:8080/?token=…`. A page opened without the token,
or a page from any other site in the same browser, cannot reach the robot.
`--token` sets a fixed token so you can bookmark the address. By default
the server listens only on this PC (127.0.0.1). `--host 0.0.0.0` makes it
reachable from the network, and then anyone with the address, token
included, can move the robot.

**Safety**

- **V3 controller protocol (Stop() -10000, MovJ(joint=...) -30001):**
  older Nova 2 firmware, the one x-trainer drives, takes motion commands on
  port 30003 (`JointMovJ(j1,...,j6)`) and has no `Stop()`.
  `dobot_tcp.py` detects this when it connects: if port 30003 is open it
  uses V3, so joint moves become SpeedJ / AccJ + JointMovJ and STOP becomes
  `ResetRobot()`. A straight line (MovL) becomes JointMovJ steps 1 cm
  apart along the line, each one checked like any other move. Use
  `--protocol v3` or `--protocol v4` to force one (`--sim --protocol v3`
  runs the fake controller in V3 mode). The protocol in use is printed at
  start.
- **Firmware without GetPose arguments:** some V4 firmware refuses
  `GetPose(user=0,tool=0)` with ErrorID -40001. The model check then uses
  plain `GetPose()`, which reports the tool selected on the pendant. It
  accepts tool 0 (the flange) or the 197 mm tool (x-trainer's tool 1); any
  other tool makes arming refuse, with a message saying so.
- **Arming:** nothing moves until **Arm** is switched on. Arming needs the
  robot enabled and idle, a table height, and the model check: GetPose
  must agree with the FK of GetAngle within 3 mm.
- **One gate for every step:** every motion and every gripper action the
  app makes, including each move of a pick, passes one check right before
  it is sent: armed, not stopped, robot enabled and idle, feedback fresh.
  STOP holds the same lock, so a move is either sent before the Stop() or
  not at all.
- **STOP:** STOP (or Esc), or switching Arm off during a task, sends
  Stop() and *latches*: nothing else moves, not even the rest of a pick,
  until you arm again. STOP also works when the command connection is
  down; it disarms first, then tries to reconnect and send Stop().
- **Every target is checked:**
  - joint limits
  - the whole path above the table: the joint path for MovJ, sampled at
    least every 1° (the planners do the same), and IK every 5 mm on one
    branch for MovL
  - that the arm hasn't moved since the check
  - the speed cap
  - only one motion at a time
- **Automatic disarm:** the app disarms itself when the robot is no longer
  its to move:
  - disabled, alarm, collision or pause
  - drag mode
  - jogged from the pendant, or moved by another program
  - its feedback or command connection is lost
- **Deadman:** a running task is stopped when the last browser closes or
  reloads the page, or when no browser has been heard from for 3 s. An
  open page checks in twice a second.
- **Pick and Detect:**
  - **Pick** opens the gripper before it comes down, and is refused while
    the gripper holds something.
  - **Detect** needs `--mqtt` (or `--fake-vision` in simulation). STOP ends
    its wait, and a pose is refused if the arm moved while the camera was
    looking.
- **What the page shows:** if the page stops receiving the robot's state,
  it says "the picture is not live", draws the robot see-through and
  disables every button except STOP. On a phone, STOP floats at the bottom
  of the screen.
- **Simulation:** the fingers only close on the object if the tool tip
  really is at it. A pick that misses goes back up and home, as on the
  robot.

### Testing the vision agent: the tomato from FoundationPose's numbers

Press **Camera** (top right of the 3D view) to look through the wrist
camera. A panel, **Tomato from FoundationPose**, appears at the bottom
left. Type the object's pose as FoundationPose gives it, relative to the
camera:
- **x, y, z** in meters (x right, y down, z along the view);
- **yaw** in degrees, a turn about the camera's view axis.

The tomato moves as you type, placed relative to the camera on the arm
where the arm is now. Seen through the camera view it should land where
the real tomato is in the real camera image. Under the inputs the panel
shows:
- where the tomato is in the base frame;
- how high its centre is above the table. A 60 mm tomato on the table
  should be about 30 mm above it, so the gap shows how far off the vision
  is.

A value behind the camera (z ≤ 0), over 2 m (typed in mm), or entered
while the arm is moving is shown there as an error. Leaving the Camera
view hides the panel.

**Live:** start the vision script with `TWIN_STREAM=1`
(`TWIN_STREAM_HZ`, default 5) and it sends every tracked pose to the twin
as an `idle` command (below), so the twin's object follows the real one
while the arm is still. Nothing moves the arm.

### MQTT commands from the vision agent (EMQX)

Started with `--mqtt`, the twin also listens on `manipulator/command`
(`MQTT_COMMAND_TOPIC`) for commands like:

```json
{"state": "start", "product": "tomato", "action": "pick",
 "position": [0.02, -0.03, 0.30], "orientation": [0.0, 0.0, 0.383, 0.924]}
```

| state | action | what happens |
|---|---|---|
| `idle` (also `idal`) | — | Nothing moves. With a position, the object moves (and turns) there in the twin. |
| `start` | `pick` | The object goes to `position` (if given), then the arm picks it and holds it at home. |
| `start` | `place` | The held object goes down at `position`'s x, y on the table, then the arm goes home. |
| `stop` | — | STOP: the arm stops and the twin disarms. A bare `stop` payload works too. |

- **position:** `[x, y, z]` in the camera frame, meters. It also accepts:
  - 7 numbers `[x, y, z, qx, qy, qz, qw]`, position and quaternion in one;
  - 4 numbers `[x, y, z, w]`, with w a turn about the view axis in degrees;
  - `"x,y,z"` or `{"x": …, "y": …, "z": …}`.

  It is taken from where the arm is when the command arrives.
- **orientation:** FoundationPose's quaternion `[qx, qy, qz, qw]`, scalar
  last (`quaternion` and `{"x", "y", "z", "w"}` work too). Instead of
  position and orientation you can send FoundationPose's 4x4 as
  `"T_camera_object"`.
- **symmetric** (optional): `true` makes the grasp ignore the rotation.
  The default is `true` without an orientation and `false` with one.
- **product:** a label shown in the twin and sent back in the replies.
- **request_id** (optional): echoed in the replies.
- **Replies:** they go to `manipulator/status`: `{"state": "accepted" |
  "refused" | "done" | "failed" | "stopped" | "updated" | "idle",
  "action", "product", "detail", "stamp"}`.
- **Safety:** commands that move the arm need **Arm** switched on in an
  open twin page, the same as from the page itself, and the deadman still
  applies. Without them a command is `refused`. `stop` always works.

To send a command by hand (FoundationPose's printed numbers: position,
then quaternion):

```
python3 mqtt_send.py idle --position 0.02 -0.03 0.30 --orientation 0 0 0.383 0.924   # show it in the twin
python3 mqtt_send.py start pick --position 0.02 -0.03 0.30 --orientation 0 0 0.383 0.924 --product tomato
python3 mqtt_send.py start pick --position 0.02 -0.03 0.30 --round                   # a tomato: rotation ignored
python3 mqtt_send.py start place --position -0.10 0.05 0.32
python3 mqtt_send.py stop
python3 mqtt_send.py --json '{"state":"idle","position":[0.02,-0.03,0.30],"orientation":[0,0,0.383,0.924]}'
```

`mqtt_send.py` prints the replies. You can also paste the same JSON into
the EMQX dashboard's WebSocket client or MQTTX: publish to
`manipulator/command` at QoS 1, and subscribe to `manipulator/status`.

**EMQX connection:** the agent, the twin, the vision script and
`mqtt_send.py` all read the same environment variables:

```
export MQTT_BROKER_HOST=192.168.5.10       # your EMQX (or xxxx.emqxsl.com for EMQX Cloud)
export MQTT_BROKER_PORT=1883               # 8883 = TLS
export MQTT_USERNAME=...                   # if EMQX authentication is on
export MQTT_PASSWORD=...
export MQTT_TLS=1                          # on by itself on port 8883
export MQTT_CA_FILE=/path/emqxsl-ca.crt    # EMQX Cloud's CA (from its console), for TLS
export MQTT_TRANSPORT=websockets           # only for the WebSocket listener: port 8083 (8084 TLS), path /mqtt
```

- **Client IDs:** each client gets a random client ID suffix, because EMQX
  disconnects the older of two clients that share an ID.
- **Wrong credentials:** if EMQX refuses the connection, the error names
  its reason, e.g. "Not authorized" or "Bad user name or password". Also
  check the ACL if authorization is on: these clients publish and
  subscribe to `manipulator/#`.

**Model:**
- **Arm:** the URDF chain the browser draws matches `manipulator_kinematics`
  within 0.016 mm, which `test_digital_twin.py` checks.
- **Gripper:** there is no CAD of your gripper, so the twin draws a simple
  parallel gripper at your 197 mm tool length. The camera box sits where the
  camera table mounts it.
- **Offline use:** three.js 0.160 (MIT) is in `twin_web/vendor`, so the
  cell PC needs no internet. The browser needs WebGL.

## Real-time pick on the robot

```
 vision PC                          MQTT broker                    arm PC
 foundationpose_mqtt.py  <---------- ready -------------  pick_runner.py --TCP 29999--> Nova 2 controller
 (RealSense, YOLO, SAM,  ------ T_camera_object ------->        |       --RS485-----> Feetech gripper
  FoundationPose)                                               +--- status -------> manipulator/status
```

The controller plans every motion itself. `pick_runner.py` only sends the
joint targets the IK chose and checks each one is reached before going on.
One cycle:

1. The arm is at the detection pose (home) and holding still, with the gripper open.
   At start the runner moves it there with a table-checked MovJ.
2. It publishes **ready**. The vision PC finds the tomato and tracks it. Once the
   tomato has held still for **2.5 s**, the PC publishes `T_camera_object`.
3. The runner reads the joints, computes
   `T_base_object = FK_camera(joints) @ T_camera_object`, and plans the whole
   pick, place and way home with the table check before anything moves.
4. **MovJ** to the pre-grasp 10 cm above the tomato, then **MovL** straight down
   to the grasp.
5. **Measure** before gripping. The joints (GetAngle → FK) must put the tool tip
   within 2 mm and 1° of the grasp. The controller's own flange position
   (GetPose) must agree within 5 mm (`--reach-tol-mm` plus the 3 mm model
   tolerance).
6. **Close** the gripper. The fingers have to stop on the tomato. If they close
   all the way, it opens, goes back up and home, and reports "failed".
7. **MovL** straight up. With `--place X Y` it then does MovJ above the place
   spot, MovL down, measures, opens, and does MovL back up.
8. **MovJ** home, publish "done", and start the next cycle (`--cycles`, 0 = forever).

**Setup**

- **Arm PC:** run `pip install -r requirements.txt`. `scservo_sdk/` is in the repo.
  Put your user in the `dialout` group for /dev/ttyUSB*. The robot IP
  defaults to 192.168.5.1, as in your `nova2_control.py`. The gripper
  follows x-trainer's pairing: 192.168.5.1 → left (ID 21 on /dev/ttyUSB1),
  192.168.5.2 → right (ID 22 on /dev/ttyUSB0). For any other IP pass `--arm`.
  `--gripper-port` and `--gripper-id` override it. The runner prints the
  gripper it uses.
- **Vision PC:** copy `foundationpose_mqtt.py` and `mqtt_link.py` into the
  folder with `FoundationPose/`, `meshes/`, the SAM checkpoint,
  `precise_6dof_pose.py` and the camera modules. Install `paho-mqtt`, then run
  `python3 foundationpose_mqtt.py`. It takes frames from router.py
  (remote_camera, 127.0.0.1:5601); add `--local-camera` for a RealSense
  plugged into the vision PC. SPACE makes it look without a ready message,
  for bench tests. A running pick_runner.py will act on that pose.
- **Both:** set the same broker variables as your scheduler stack:
  `MQTT_BROKER_HOST`, `MQTT_BROKER_PORT`, `MQTT_USERNAME`, `MQTT_PASSWORD`.
  Topics default to `manipulator/ready`, `manipulator/object_pose` and
  `manipulator/status`. Change them with `MQTT_READY_TOPIC`, `MQTT_POSE_TOPIC`
  and `MQTT_STATUS_TOPIC`. Keep credentials in `.env` (git-ignored) and
  export them before running, e.g. `set -a; . ./.env; set +a`. The scripts
  read only environment variables (broker default localhost:1883).

**Messages** (JSON):

```
ready  {"request_id": "3f9a01c2", "joints": [-180, 0, -90, 0, 90, 90], "stamp": ...}
       (a plain "ready" works too:  mosquitto_pub -t manipulator/ready -m ready)
pose   {"request_id": "3f9a01c2", "object": "tomato", "origin": "center", "symmetric": true,
        "diameter": 0.062, "settle_s": 2.5, "frame": "camera",
        "T_camera_object": [[1,0,0,0.03],[0,1,0,-0.02],[0,0,1,0.38],[0,0,0,1]], "stamp": ...}
status {"request_id": "3f9a01c2", "state": "waiting|picking|holding|placed|done|failed|finished|stopped",
        "detail": "...", "stamp": ...}
```

When the runner exits it publishes "finished" (all cycles ran; detail
"N cycles, M failed") or "stopped" (an error or Ctrl+C stopped the robot).

The runner sends ready again every 5 s until a pose arrives. This covers a
vision PC that started late or a pose message that was lost. If the vision
side is still working on that request, it ignores the repeat. If it
answered less than 10 s ago, it sends the same pose again. If the answer is
older, it detects again, because the tomato may have moved. A pose with
another `request_id` is ignored. A pose with none (an answer to a plain
"ready", or SPACE on the vision PC) is accepted, so don't press SPACE while
the arm is waiting. An invalid ready message is logged and skipped.

**What is checked, and what happens if it fails**

| When | Check | If it fails |
|---|---|---|
| Start | GetPose (tool 0) matches the FK of GetAngle within 3 mm | Refuses to run: the kinematic model doesn't match the controller (see the next section) |
| Start | If the arm is not at home, the MovJ home is checked against the table. A dry run without `--table-z` only warns. Then the arm moves home | Refuses to run: jog the arm closer to home first |
| Every cycle | RobotMode is 5 (enabled and idle) before ready is sent | Stops |
| Waiting for the pose | `--pose-timeout S` (default: wait forever) | "failed", next cycle |
| Pose received | The joints did not change while the camera looked | Stops |
| Before moving | Every waypoint IK-solved above the table, every 2 cm on the lines down and up, and the joint paths between them | "failed"; the arm does not move |
| Every move | Done = its queue ID is current, RobotMode 5, and at the target (MovJ: joints within 0.3°; MovL: tool tip within 2 mm and 1°). Any of these → error: an alarm, a collision, a disable, stopping short, no joint moving 0.05° for 60 s (a pause counts, reading noise doesn't), or 30 min in all. A slow move is fine as long as it keeps moving | Stop() is sent, then the script exits |
| Before closing or opening | GetAngle → FK tool tip and GetPose flange are both at the target | Stops |
| After closing | The fingers stopped short of fully closed (30 servo counts) | Opens, MovL up, MovJ home, "failed" |

**Commands**

```
python3 dobot_tcp.py --ip 192.168.5.1               # read only: mode, joints, pose, model check, alarms
python3 gripper.py --arm left open                  # just open it (or: close)
python3 gripper.py --arm left                       # test: open, close, open; says if it holds something
python3 pick_runner.py --dry-run --fake-vision demo # fake robot, gripper and vision
python3 pick_runner.py --dry-run                    # fake robot and gripper, your broker and vision PC
python3 pick_runner.py --table-z 0.018 --speed 10   # first real pick: slow; holds it at home, ends
python3 pick_runner.py --table-z 0.018 --place -0.25 0.40 --cycles 0   # pick and place, forever
python3 pick_runner.py --robot-ip 192.168.5.2 --table-z 0.018          # right arm, its gripper (ID 22)
```

Notes:

- **`--table-z`** (required on the robot): the table surface in the base
  frame, in meters. z = 0 is the robot base's mounting face, so a robot
  standing on the table has its table at about 0, not below it. 0.018 in
  the examples is what one setup measured; measure yours (see "Table
  height" below). A value that isn't between −1 m and 5 cm below the tool
  tip at home is refused, which catches a reading typed in mm, and a
  value below −0.02 (table under the base) needs `--pedestal`. Only
  `--dry-run` may leave it out; it then estimates the table as the
  tomato's centre minus half its diameter. The real robot needs the
  measured value, because an estimate from the same camera pose as the
  grasp would carry the same error.
- **Table height:** the table check is only as good as `--table-z`:
  everything is kept above it (arm 5 cm, gripper 5 mm), and nothing else
  knows where the table is. To measure it, put the fingertips on the table
  with the gripper pointing down (drag mode, or jog slowly), then read
  **Lowest point** in the twin's tool tip panel, or run
  `python3 dobot_tcp.py --ip 192.168.5.1` ("lowest ... start with
  --table-z ..."). Measuring with the same model the check uses also takes
  care of fingers a little longer than the modelled 197 mm. Measure at two
  or three places across the table; if they differ by more than a few mm
  the table is not level in the robot's frame, so use the highest. In the
  twin, **Above table** shows how far the lowest point is above the table
  height you started with: with the fingertips on the table it should
  read about 0. Arming refuses if the arm is already below that height.
  Also set the controller's collision detection on the pendant, as a
  second line of defence.
- **Grasp point:** FoundationPose's tomato frame is its centre, because the mesh is
  centred on its bounding box. The tool tip goes to the centre
  (`--grasp-height 0`). For an object whose frame is its bottom
  (`"origin": "bottom"`), the default is 2 cm above it.
- **Gripper turn:** with `LOCK_ROTATION = True` the pose has no usable yaw.
  The message says `"symmetric": true`, and the wrist stays turned as it is at
  home. `--grasp-yaw` adds a turn.
- **Speeds:** `--speed` is the MovJ speed and acceleration in %. `--line-speed`
  is the same for MovL. Both are multiplied by the controller's global speed,
  which stays as set on the pendant unless you pass `--speed-factor N`.
  Defaults: `--speed 20`, `--line-speed 10`. Use 10 for the first runs.
- **Other options:** `--approach` sets the pre-grasp height above the grasp
  (0.10 m). `--reach-tol-mm` sets the tool tip tolerance (2). `--pose-timeout
  S` gives up waiting for the vision after S seconds and reports "failed".
  `--fake-vision FILE` answers ready with this 4x4 `T_camera_object` instead
  of a vision PC. `python3 pick_runner.py --help` lists them all.
- **Squeeze:** `--grip-torque` sets the servo torque limit (0..1000). The default
  300 is x-trainer's value. Go lower if it squashes the tomato.
- **Without `--place`:** the arm ends at home holding the tomato, so only
  `--cycles 1` is allowed.
- **Ctrl+C** sends Stop() to the controller. Keep the emergency stop within
  reach anyway.
- **Detection pose:** home by default. `--home J1 … J6` sets another one,
  e.g. with the base turned 90°. The camera must look down at the table from
  there.

## Correction to your DH table (J1, J3 and J5 offsets)

Your link lengths and twists are right, but three θ offsets did not match the
real Nova 2. I compared the model with Dobot's official Nova 2 description,
`dobot_description/urdf/nova2_robot.urdf` in
[Dobot-Arm/TCP-IP-ROS-6AXis](https://github.com/Dobot-Arm/TCP-IP-ROS-6AXis).
Dobot's driver feeds the controller's joint angles into it unchanged.

| Pose (servo deg) | Dobot's Nova 2: flange | Your offsets: flange |
|---|---|---|
| home | (−0.345, **+0.1175**, 0.4154) m | (−0.345, **−0.1175**, 0.4154) m, 235 mm off |
| home, J2 +30° | rises to z 0.562 | drops to z 0.217 |
| home, J3 +30° | forearm lifts, z 0.600 | forearm drops, z 0.255 |

With your offsets the model's home pose is the mirror twin of the real one.
From the side it looks identical, but the shoulder offset is on the other
side and J2–J4 turn the opposite way from home. Over random joint angles the
model was 570 mm off on average, so the IK would have sent the arm to the
wrong place. Your `effective_transformation.py` "measured" point has
Y = +0.1076, on Dobot's side.

`manipulator_kinematics.py` now uses θ1 = q1 + 180°, θ3 = q3 − 90° and
θ5 = q5 − 90° in all three tables (yours: q1, q3 + 90°, q5 + 90°). That
matches Dobot's model within 0.02 mm. Home keeps the same servo angles, the
same flange orientation and the same tool and camera offsets. If you keep
using your own scripts (`effective_transformation.py`,
`tooltip_transformation.py`, `camera_tranformation.py`, `build_dh_params` in
`joint_mapping.py`), make the same three changes there.

**Confirm with one reading on the robot:** move to home, select no tool (tool
0), and read the pose on the pendant (or `GetPose`).

| Pendant reads (mm) | Meaning |
|---|---|
| X ≈ −345, Y ≈ +117.5, Z ≈ 415.4 | The corrected model matches your controller. |
| X ≈ +345, Y ≈ −117.5, Z ≈ 415.4 | Same arm, but the controller's base frame is turned 180°. Set the J1 offset back to 0. The joint angles the IK commands don't change. |
| X ≈ −345, Y ≈ −117.5, Z ≈ 415.4 | Your original offsets were right after all. Set J1, J3 and J5 back to 0, +90° and +90°. |

## Pose → joint angles (`ik_from_pose.py`)

Give the target pose of any of the three frames: `--frame tool` (default),
`flange` or `camera`. Give it as a 4x4 file (`--matrix`: `.npy`, or a text
file of 16 numbers, `np.savetxt` output or the matrix pasted as the scripts
print it, brackets and all) or as a position (`--xyz`, plus `--yaw` in
degrees to turn it about the vertical, + = counter-clockwise seen from
above). A rotation rounded to a few decimals is snapped to the nearest exact
rotation; a mirror image (determinant −1) is refused. Positions are in
metres. An unreachable target exits with code 2. The script converts it to the tool tip pose in the base
frame, `T_base_tool = T_base_frame @ inv(T_FLANGE_frame) @ T_FLANGE_TOOL`,
solves the IK, and checks the answer with FK of the frame you gave.

**In the base frame:** `--xyz` points the frame straight down, the way it
points at home. At home, all three frames give the home joints:

| Frame | Pose at home (x, y, z) m | Joint angles (servo deg) |
|---|---|---|
| flange | (−0.345, 0.1175, 0.4154) | `[-180, 0, -90, 0, 90, 90]` |
| tool | (−0.345, 0.1175, 0.2184) | `[-180, 0, -90, 0, 90, 90]` |
| camera | (−0.4215, 0.1175, 0.3654) | `[-180, 0, -90, 0, 90, 90]` |

Reproduce a row with, for example,
`python3 ik_from_pose.py --frame camera --xyz -0.4215 0.1175 0.3654`, or with
`--frame camera --matrix T_base_camera.txt` holding the matrix
`fk_from_joints.py` prints at home.

**Relative to an object:** add `--object FILE`. By default that file holds
`T_camera_object` from FoundationPose, taken at `--detect` joints (default
home). Pass `--object-in base` if it already holds `T_base_object`. Then
`--matrix` is `T_object_frame` and `--xyz` is a position in the object
frame. With `--xyz` the frame's z axis runs along the object's −z, so it
points at the object's origin only when x = y = 0 and z > 0. Its x axis is
the object's x, turned by `--yaw` about the object's +z. If your object's
+z doesn't point away from the table, use `--matrix`. For example:

- `--frame camera --xyz 0 0 0.30`: camera 30 cm out along the object's +z
  (above it when +z is up), looking back at it
- `--frame tool --xyz 0 0 0.02`: tool tip 2 cm above the object's origin
- `--frame flange --matrix T_object_flange.txt`: any flange pose in the
  object frame

In code:

```python
import numpy as np
from ik_from_pose import facing_pose, object_in_base, solve_frame

T_camera_object = np.loadtxt("T_camera_object.txt")   # FoundationPose 4x4, image taken at home
T_base_object = object_in_base(T_camera_object)        # not at home: q_detect_servo_deg=[...]
T_object_camera = facing_pose(0, 0, 0.30)              # = --frame camera --xyz 0 0 0.30
T_base_tool, joints = solve_frame(T_base_object @ T_object_camera, "camera",
                                  current_servo_deg=[-180, 0, -90, 0, 90, 90])
print(joints[0])                                       # send this one to your driver
```

`down_pose(frame, x, y, z, yaw)` builds the base-frame `--xyz` pose the same
way.

## Conventions

- **Units and frame:** poses are 4x4 matrices in the arm's base frame (frame
  0 of the DH tables, the same as Dobot's base frame), in metres.
- **Model angles (`q`):** the "visual" angles of your joint mapping, in
  radians, all zero at home.
- **Servo angles (`*_servo_deg`):** what the controller reports and accepts,
  in degrees. Home is `[-180, 0, -90, 0, 90, 90]`.

## 1. Object relative to the base

Detection happens at home, where the wrist camera looks straight down at the
table:

```
T_base_object = T_base_camera(home) @ T_camera_object
```

`T_base_camera` is the FK of your camera table. `T_camera_object` is your
detector's output: either a 4x4 matrix, or `rvec, tvec` from `cv2.aruco` /
`solvePnP` passed through `pose_from_rvec_tvec`. `tvec` is in metres when
the marker size is given in metres. The base as seen from the object is the
inverse, `plan.T_object_base`.

## 2. Tool-tip IK that never goes below the table

- **Grasp:** straight down (tool z into the table), with tool x along the
  object's yaw. The 180°-turned grip is tried too, and the one with less
  joint travel wins.
- **Approach:** from a pre-grasp 10 cm above, the tip goes down and back up
  the vertical line through the object. It moves through joint waypoints
  every 2 cm, so it stays within about 2 mm of that line, and under 0.3 mm
  in the area the camera sees from home. The worst cases are near full reach.
  A single joint move
  would bow up to 13 mm sideways.
- **Every waypoint:** solved with the closed-form IK. A solution is dropped
  if it breaks the joint limits, or if the elbow, wrist, flange or camera
  comes within 5 cm of the table, or the gripper tip within 5 mm. The joint
  moves between waypoints get the same check, and the descent must stay on
  one IK branch (no elbow or wrist flip above the object).
- **Choice:** of the safe solutions, the one with the least joint travel
  from the current pose.
- **No safe solution:** `IKError` is raised and names the part that was too
  low, e.g. `gripper 2.0 cm too low`.

**Table height:** `table_z` in the base frame. With `table_z=None` it is the
plane the detected object sits on, since the ArUco markers lie on the table.
That is for planning offline. On the robot, pass the measured height
(`pick_runner.py` requires `--table-z`). A table height taken from the same
camera pose as the grasp would share its error.
The table is treated as an endless floor at that height. Pass
`table_x_range` / `table_y_range` if the arm also works beside the table,
lower than its top.

## Using it in your control code

```python
from manipulator_agent import ManipulatorAgent, pose_from_rvec_tvec
from manipulator_kinematics import IKError, joint_map

agent = ManipulatorAgent(table_z=TABLE_Z)   # measured once, see --table-z; None (the object's plane) is for offline planning only

# 1. Move the arm to home (joint_map.home_servo_deg) and detect the object there.
T_camera_object = pose_from_rvec_tvec(rvec, tvec)

# 2. Plan, then execute with your driver.
try:
    plan = agent.plan_pick(T_camera_object)
except IKError as e:
    print("Not safe to pick:", e)
else:
    print("Object in base frame:\n", plan.T_base_object)
    for wp in plan.waypoints:
        robot.joint_move(wp.q_servo_deg)  # your driver, e.g. Dobot JointMovJ
        if wp.gripper == "open":
            gripper.open()
        elif wp.gripper == "close":
            gripper.close()
```

`robot.joint_move` and `gripper` stand for a driver. `pick_runner.py` does
this on the real robot with `dobot_tcp.py` and `gripper.py`, sending only
the pre-grasp (MovJ) and the grasp (MovL): the controller draws the straight
line down itself, and the planner's waypoints along it are what proves that
line stays on one IK branch above the table.

For any other tool target (a place pose, say), use the IK directly:

```python
from manipulator_kinematics import TableConstraint, solve_ik

q_servo_deg = solve_ik(T_base_tool, current_servo_deg, TableConstraint(z=table_z))
```

`plan_pick` parameters: `q_detect_servo_deg` (the pose the image was taken
at; default home) and `q_start_servo_deg` (the pose the arm moves from;
default the detection pose).

`ManipulatorAgent` settings (metres unless noted):

| Parameter | Default | Meaning |
|---|---|---|
| `table_z` | `None` | Table surface height in the base frame; `None` = the object's plane |
| `link_clearance` | `0.05` | Elbow / wrist / flange / camera above the table |
| `tip_clearance` | `0.005` | Gripper tip above the table |
| `table_x_range`, `table_y_range` | `None` | Table footprint in the base frame; `None` = endless |
| `grasp_height` | `0.02` | Tool tip above the object's origin at the grasp |
| `approach_height` | `0.10` | Pre-grasp / lift height above the grasp |
| `descent_step` | `0.02` | Waypoint spacing on the way down and up |
| `grasp_yaw_offset_deg` | `0.0` | Tool x relative to the object's x, so the fingers close across it |
| `max_branch_jump_deg` | `90.0` | Largest joint change between descent waypoints |

## Tool and camera offsets

The tool table is your updated one: d5 0.12 and d6 0.285, with no J6
offset. The tip sits on the flange axis, 0.197 m out from the flange, so the
fixed flange→tool transform is exact for every joint angle.

The camera table still puts a 76.5 mm sideways offset into `d5` (0.1965).
That is only right while J6 is at home: the camera turns with the flange,
so at J6 = 90° the table's camera point is 108 mm from the real one.
`manipulator_kinematics.py` therefore reads the camera table once at home
to get a fixed flange→camera transform, which matches the table at J6 = 0
and stays right as J6 turns.

Keep editing the DH rows in `manipulator_kinematics.py`; the offsets follow
them.

**Your new tool table turns every grasp 90°.** It has no J6 offset, so tool
x is now the flange's x axis; your old table's tool x was the flange's y.
The planners put tool x along the object's x, so for the same detection the
gripper now ends up 90° from before. Check on the robot which flange axis
your fingers close along. If it's the flange's y axis, use
`ManipulatorAgent(grasp_yaw_offset_deg=90)`, and in `pick_from_pose.py` set
`T_OBJECT_GRASP = pose(rot_x(np.pi) @ rot_z(np.pi / 2), [0, 0, 0.02])`.

## The FK → IK round-trip test

`test_ik_random.py` draws random joint angles inside `SERVO_LIMITS_DEG`. For
each set it computes the tool-tip transformation with FK, solves IK for that
transformation, and checks three things:

- **Same angles back:** with the original angles as the reference (how the
  robot chooses, using its current angles), the nearest IK solution must
  equal them within 1e-6°.
- **Not just an echo:** from an unrelated random reference, the original
  angles must still be among the solutions. An IK that merely returned its
  reference would fail here.
- **Same pose back:** FK of every IK solution must reproduce the
  transformation within 1e-10 m and 1e-8°. The worst seen is about 1e-15 m.

It runs three sections:

- **Random:** joint states drawn across the whole joint range.
- **Limits:** each joint exactly at its lower or upper limit.
- **Singular poses:**
  - wrist straight (J5 = 0° or 180°, J4 and J6 parallel)
  - elbow in line (J3 = 0°)
  - shoulder (the wrist centre on the radius-d4 cylinder around J1)
  - states right next to each of these
  - a reference whose J6 doesn't match

  At and right next to a singularity, the pose no longer pins every joint
  down on its own, so those cases are judged by pose. At the exact wrist
  singularity the joints are checked too, because J6 then comes from the
  reference.

The script exits with code 1 on any failure.

I mutation-tested it with 12 deliberately injected IK bugs, each run on its
own copy:

- **Caught (10):** an IK that echoes its reference, returning only the
  nearest solution, the old `acos` wrist bug, ignoring the J6 reference at
  the wrist singularity, no J6 spread there, a strict joint-limit compare,
  no shoulder tolerance, a dropped elbow branch, a dropped wrist branch, and
  a 0.1 mm d4 mismatch between FK and IK.
- **Unwrapped output:** harmless to the round trip, and caught by
  `test_manipulator.py`.
- **No J6 clamp at the singularity:** changes nothing the test can see,
  because the J6 spread already covers the reachable range.

The round trip proves the IK inverts the FK model. Whether the model matches
the robot is checked by `test_manipulator.py` against Dobot's Nova 2 model,
and by the pendant reading above.

## Check on the robot before the first real pick

- **Pendant reading at home:** see the correction section above.
- **Joint limits:** `SERVO_LIMITS_DEG` holds the official Nova 2 limits (J2
  and J3 ±150°, the others ±359.8°). Tighten them if your controller's soft
  limits are narrower.
- **Camera frame:** `T_camera_object` must be in the camera table's frame,
  with z as the view axis. If your detector's x/y are turned relative to it,
  change that table's J6 offset.
- **Camera offset:** the camera table still has the 76.5 mm sideways offset
  your old tool table had. If the camera is centred on the flange too, set
  its d5 back to 0.12.

## Tests of the real-time side

`test_pick_runner.py`, `test_vision_mqtt.py` and `test_digital_twin.py`
need no robot, gripper, camera, GPU or broker. The browser part of
`test_digital_twin.py` runs when Playwright is installed
(`pip install playwright`) and is skipped otherwise. I mutation-tested them: I injected one bug at a
time, each on its own copy, and checked whether the tests failed.

- **`test_pick_runner.py` caught 21 of 22.** Caught:
  - returning before the move ends (no queue-ID check)
  - ignoring a collision or a stop short of the target
  - timing out on a slow move that is still moving
  - waiting forever on a stuck robot whose readings jitter
  - letting NaN past the joint limits
  - a gripper that always reports "holding"
  - three `PoseGate` timing bugs
  - accepting a mirrored pose or an unusable request ID
  - skipping the lift after a missed grip
  - skipping the "arm moved" check or the model check
  - the table estimate ignoring the diameter
  - descending with MovJ to the wrong joints
  - allowing the robot to run without `--table-z`, or with a table height
    in mm
  - exit code 0 after a failed cycle
  - an unnamed place-spot error

  The miss changes nothing for a top-down grasp: it placed the tool, rather
  than the object, at the place spot, and for a top-down grasp they are the
  same point.
- **`test_vision_mqtt.py` caught 7 of 8.** Caught:
  - not restarting the 2.5 s when the tomato is lost
  - detecting before a ready
  - counting a held (no-depth) pose
  - dropping the request ID
  - a shorter settle time
  - not sending the pose again when the arm repeats an answered ready
  - re-sending a pose that is too old

  The miss changes nothing either: it skipped a registration reset that the
  next ready does anyway.
