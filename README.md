# Manipulator control: object pose and table-safe tool-tip IK

Pick planning for the 6-DOF arm, built on your three DH tables: the flange
with no end effector, the gripper tool tip, and the wrist camera. It is
plain Python and needs only numpy. It uses no ROS2 or other middleware.

| File | What it does |
|---|---|
| `manipulator_kinematics.py` | Your DH tables (three offsets corrected, see below) and `JointMap` (servo ↔ model angles), FK for flange / tool / camera, closed-form IK (up to 8 solutions), joint limits, and the table check. |
| `manipulator_agent.py` | `T_base_object` from a camera detection, then a top-down pick plan (pre-grasp → descend → grasp → lift) whose every waypoint and move stays above the table. Also a command-line demo. |
| `pick_from_pose.py` | Full 6-DoF object pose (FoundationPose 4x4, position + orientation) → approach and grasp joint waypoints along the object's own axes; checked against the table (waypoints and the moves between them) when you pass `--table-z`. |
| `fk_from_joints.py` | Joint angles (servo deg, given or random) → flange, tool tip and camera poses; `--steps` prints every joint transform. |
| `ik_from_pose.py` | Pose of the tool tip, flange or camera (4x4 or x y z), in the base frame or relative to an object → tool tip pose in the base frame and joint angles, each checked with FK. See "Pose → joint angles" below. |
| `test_manipulator.py` | Offline checks: FK against Dobot's official Nova 2 model and the DH tables, IK round trip, table constraint, full detect → pick plan. |
| `test_ik_random.py` | Random joint angles → FK (tool tip transform) → IK → joint angles, compared with the originals. Also covers joints exactly at their limits and every singular pose. |

```
pip install -r requirements.txt
python3 pick_from_pose.py --pose pose.txt --table-z -0.05   # FoundationPose 4x4 -> joint angles
python3 fk_from_joints.py --joints -180 0 -90 0 90 90   # joint angles -> flange / tool / camera poses
python3 ik_from_pose.py                          # home pose of tool / flange / camera + random points
python3 ik_from_pose.py --frame camera --matrix T_base_camera.txt
python3 ik_from_pose.py --xyz -0.40 0.10 0.05 --yaw 30 --table-z -0.05
python3 ik_from_pose.py --object T_camera_object.txt --frame camera --xyz 0 0 0.30
python3 test_ik_random.py                        # 1000 random poses, first 3 printed in full
python3 test_ik_random.py --n 10000 --seed 42 --show 5
python3 test_manipulator.py
python3 manipulator_agent.py 0.02 -0.03 0.30 20      # object x y z [m] and yaw [deg] in the camera frame
python3 manipulator_agent.py 0.02 -0.03 0.30 20 --table-z -0.05
```

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
The table is treated as an endless floor at that height. Pass
`table_x_range` / `table_y_range` if the arm also works beside the table,
lower than its top.

## Using it in your control code

```python
from manipulator_agent import ManipulatorAgent, pose_from_rvec_tvec
from manipulator_kinematics import IKError, joint_map

agent = ManipulatorAgent(table_z=None)   # None: table = plane the markers lie on

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

`robot.joint_move` and `gripper` are placeholders for your own driver.
Command joint moves, which are the paths the table check validated.

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
