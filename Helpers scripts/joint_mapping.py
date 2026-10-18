import numpy as np


class JointMap:
    """
    Maps real servo angles (what the Dobot reports) <-> visual/model angles
    (what goes into the DH table).

        visual = direction * (servo - home_servo)
        servo  = home_servo + direction * visual

    At the home pose every visual angle is 0.
    direction[i] = +1 -> same rotation sense as the servo
    direction[i] = -1 -> flipped
    """

    def __init__(self, home_servo_deg, direction=(1, 1, 1, 1, 1, 1)):
        self.home_servo_deg = np.asarray(home_servo_deg, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        assert self.home_servo_deg.shape == (6,)
        assert self.direction.shape == (6,)
        assert np.all(np.abs(self.direction) == 1), "direction must be +1 or -1"

    def servo_to_visual(self, servo_deg):
        """servo angles [deg] -> visual angles [rad] for the DH table"""
        servo = np.asarray(servo_deg, dtype=float)
        return np.radians(self.direction * (servo - self.home_servo_deg))

    def visual_to_servo(self, visual_rad):
        """visual angles [rad] -> servo angles [deg] to command the robot"""
        visual = np.asarray(visual_rad, dtype=float)
        return self.home_servo_deg + self.direction * np.degrees(visual)


# ----------------------------------------------------------------------
# EDIT HERE
# ----------------------------------------------------------------------
joint_map = JointMap(
    home_servo_deg=[-180, 0, -90, 0, 90, 90],   # j1..j6 at your home pose
    direction=[1, 1, 1, 1, 1, 1],               # set any to -1 to flip that joint
)


def build_dh_params(q):
    """
    q: 6 visual angles in radians (output of joint_map.servo_to_visual).
    Rows are [alpha, a, d, theta], same as your original table.
    """
    theta1, theta2, theta3, theta4, theta5, theta6 = q
    return np.array([
        [np.pi / 2, 0.0,   0.2234, theta1],                    # Joint 1
        [0,         0.28,  0.0,    theta2 + np.radians(90)],   # Joint 2
        [0.0,       0.225, 0.0,    theta3 + np.radians(90)],   # Joint 3
        [np.pi / 2, 0.0,   0.1175, theta4 + np.radians(90)],   # Joint 4
        [np.pi / 2, 0.0,   0.12,   theta5 + np.radians(90)],   # Joint 5
        [0,         0.0,   0.088,  theta6],                    # Joint 6
    ])


if __name__ == "__main__":
    home = [180, 0, 90, 0, -90, -90]

    print(joint_map.visual_to_servo(np.radians(home)))

    print(np.degrees(joint_map.servo_to_visual([0, 0, 0, 0, 0, 0])))

