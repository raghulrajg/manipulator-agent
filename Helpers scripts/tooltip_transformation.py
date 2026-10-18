import numpy as np


def dh_transform(alpha, a, d, theta):
  """
  Computes the 4x4 Homogeneous Transformation Matrix for a single joint
  given standard Denavit-Hartenberg parameters:
  - alpha : link twist (radians)
  - a     : link length (meters)
  - d     : link offset (meters)
  - theta : joint angle (radians)
  """
  ct = np.cos(theta)
  st = np.sin(theta)
  ca = np.cos(alpha)
  sa = np.sin(alpha)

  T = np.array([
      [ct, -st * ca,  st * sa, a * ct],
      [st,  ct * ca, -ct * sa, a * st],
      [0,   sa,       ca,      d],
      [0,   0,        0,       1],
  ])
  return T


def forward_kinematics_6dof(dh_table):
  """
  dh_table: Nx4 numpy array where each row is [alpha, a, d, theta]

  Returns:
    T_effective : T_0^N (base to end-effector)
    transforms  : list of individual transforms T_{i-1}^{i}
    cumulative  : list of cumulative transforms T_0^{i}
  """
  T_effective = np.eye(4)
  transforms = []
  cumulative = []

  for row in dh_table:
    alpha, a, d, theta = row
    T_i = dh_transform(alpha, a, d, theta)
    T_effective = T_effective @ T_i
    transforms.append(T_i)
    cumulative.append(T_effective.copy())

  return T_effective, transforms, cumulative


if __name__ == "__main__":
  # Joint angles (degrees -> radians)
  theta1 = np.radians(0)
  theta2 = np.radians(0)
  theta3 = np.radians(0)
  theta4 = np.radians(0)
  theta5 = np.radians(0)
  theta6 = np.radians(0)

  # Standard DH parameter table (meters)
  # Columns: [alpha (rad), a (link length), d (link offset), theta (rad)]
  dh_params = np.array([
      [np.pi / 2, 0.0,     0.2234,  theta1],                      # Joint 1
      [0,         0.28,    0.0,     theta2 + np.radians(90)],     # Joint 2
      [0.0,       0.225,   0.0,     theta3 + np.radians(90)],     # Joint 3
      [np.pi / 2, 0.0,     0.1175,  theta4 + np.radians(90)],     # Joint 4
      [np.pi / 2, 0.0,     0.12,    theta5 + np.radians(90)],     # Joint 5
      [0,         0.0,     0.285,   theta6],                      # Joint 6
  ])

  T_0_6, individual_transforms, cumulative_transforms = forward_kinematics_6dof(
      dh_params
  )

  np.set_printoptions(precision=4, suppress=True)

  print("=== Individual transforms T_{i-1}^{i} ===\n")
  for i, T_i in enumerate(individual_transforms, start=1):
    print(f"T_{i-1}^{i}:")
    print(T_i)
    print()

  print("=== Cumulative transforms T_0^{i} ===\n")
  for i, T_c in enumerate(cumulative_transforms, start=1):
    print(f"T_0^{i}:")
    print(T_c)
    print()

  print("Effective Transformation Matrix (T_0^6):")
  print(T_0_6)

  position = T_0_6[:3, 3]
  rotation_matrix = T_0_6[:3, :3]

  print(f"\nEnd-Effector Position (X, Y, Z): {position}")
  print(f"\nRotation Matrix:\n{rotation_matrix}")
