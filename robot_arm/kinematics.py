import math
from .model import RobotArm # Assuming RobotArm is in model.py in the same directory

# Basic vector and matrix operations (can be replaced by NumPy later)

def matrix_multiply(A, B):
    """ Multiplies matrix A by matrix B or matrix A by vector B. """
    C = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] # For 4x4 matrices
    if isinstance(B[0], list): # Matrix * Matrix
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    C[i][j] += A[i][k] * B[k][j]
        return C
    else: # Matrix * Vector (assuming B is a 4x1 vector [x,y,z,1])
        # This path is not used by the current FK or IK, but provided for general use.
        C_vec = [0,0,0,0]
        for i in range(4):
            for k in range(4):
                C_vec[i] += A[i][k] * B[k]
        return C_vec

def create_translation_matrix(dx, dy, dz):
    return [
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ]

def create_rotation_matrix_x(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ]

def create_rotation_matrix_y(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ]

def create_rotation_matrix_z(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return [
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

def calculate_forward_kinematics(robot_arm: RobotArm, joint_angles: list[float]) -> tuple[float, float, float]:
    """
    Calculates the (X, Y, Z) position of the end effector using forward kinematics.
    Convention: For each joint, rotate then translate along the new X-axis by bone_length.
    Args:
        robot_arm: An instance of the RobotArm class.
        joint_angles: A list of joint angles in radians, one for each joint.
    Returns:
        A tuple (x, y, z) representing the position of the end effector.
    """
    if len(joint_angles) != robot_arm.num_joints:
        raise ValueError("Number of joint angles must match the number of joints in the robot arm.")

    current_transform = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]

    for i in range(robot_arm.num_joints):
        angle_rad = joint_angles[i]
        bone_length = robot_arm.bone_lengths[i]
        axis = robot_arm.joint_axes[i]

        rot_matrix = None
        if axis == 'x':
            rot_matrix = create_rotation_matrix_x(angle_rad)
        elif axis == 'y':
            rot_matrix = create_rotation_matrix_y(angle_rad)
        elif axis == 'z':
            rot_matrix = create_rotation_matrix_z(angle_rad)
        else:
            raise ValueError(f"Unsupported joint axis: {axis}")

        current_transform = matrix_multiply(current_transform, rot_matrix)

        trans_matrix = create_translation_matrix(bone_length, 0, 0)
        current_transform = matrix_multiply(current_transform, trans_matrix)

    end_effector_position = (current_transform[0][3], current_transform[1][3], current_transform[2][3])
    return end_effector_position


def calculate_inverse_kinematics(robot_arm: RobotArm, target_position: tuple[float, float, float]) -> list[float] | None:
    """
    Calculates the joint angles for a 2-link planar robot arm to reach the target_position (X,Y).
    Assumes both joints rotate around the Z-axis and the arm operates in the XY plane.
    The Z component of target_position is currently ignored.

    Args:
        robot_arm: An instance of the RobotArm class. Must be a 2-DOF planar arm with Z-axis rotations.
        target_position: A tuple (x, y, z) representing the desired position of the end effector.

    Returns:
        A list [q1, q2] of joint angles in radians if a solution is found within joint limits.
        Returns None if the target is unreachable or no solution satisfies joint limits.
        Prioritizes solutions with acos(cos_q2) >= 0 for q2 (often "elbow out").
    """
    if not (robot_arm.num_joints == 2 and
            robot_arm.joint_axes[0].lower() == 'z' and
            robot_arm.joint_axes[1].lower() == 'z'):
        raise ValueError("This IK solver is specifically for 2-link planar arms with Z-axis rotations.")

    l1 = robot_arm.bone_lengths[0]
    l2 = robot_arm.bone_lengths[1]
    x = target_position[0]
    y = target_position[1]

    # l1 and l2 should be positive, RobotArm model validation should handle this.
    if l1 <= 0 or l2 <= 0:
        # This case should ideally be prevented by RobotArm model validation,
        # but good to have a safeguard in IK if model allows non-positive lengths.
        return None

    dist_sq = x*x + y*y

    # Denominator for cos_q2
    denom_cos_q2 = 2 * l1 * l2
    if denom_cos_q2 == 0: # Avoid division by zero if l1 or l2 is zero (already checked by l1,l2 <=0)
        return None

    cos_q2_val = (dist_sq - l1*l1 - l2*l2) / denom_cos_q2

    # Clamp cos_q2_val to [-1, 1] due to potential floating point inaccuracies
    # This also handles the reachability check: if |cos_q2_val| > 1, target is unreachable
    epsilon = 1e-9 # Epsilon for floating point comparisons
    if cos_q2_val > 1.0 + epsilon or cos_q2_val < -1.0 - epsilon:
        return None # Target is unreachable (too far or too close for non-collinear case)

    # Clamp value strictly to valid acos domain after reachability check based on epsilon
    cos_q2_val = max(-1.0, min(1.0, cos_q2_val))

    # Two possible solutions for q2 (elbow "out" and "in")
    # q2_opt1 corresponds to positive acos result (typically 0 to pi)
    # q2_opt2 corresponds to negative acos result (typically 0 to -pi)
    q2_options = [math.acos(cos_q2_val), -math.acos(cos_q2_val)]

    for q2_rad_attempt in q2_options:
        s2 = math.sin(q2_rad_attempt)
        c2 = math.cos(q2_rad_attempt)

        # Calculate q1 using atan2(y,x) - atan2(l2*s2, l1+l2*c2)
        # Numerator and denominator for the second atan2 term
        atan2_term2_y = l2 * s2
        atan2_term2_x = l1 + l2 * c2

        q1_rad_attempt = math.atan2(y, x) - math.atan2(atan2_term2_y, atan2_term2_x)

        # Normalize angles to a consistent range, e.g., [-pi, pi] or [0, 2pi]
        # This helps when checking against joint limits that might also be normalized.
        # Python's % operator behavior with negative numbers needs care for normalization.
        # A common way to normalize to [-pi, pi]: angle = (angle + pi) % (2*pi) - pi
        q1_rad_normalized = (q1_rad_attempt + math.pi) % (2 * math.pi) - math.pi
        q2_rad_normalized = (q2_rad_attempt + math.pi) % (2 * math.pi) - math.pi

        q1_min, q1_max = robot_arm.joint_limits[0]
        q2_min, q2_max = robot_arm.joint_limits[1]

        limit_epsilon = 1e-7 # Epsilon for checking joint limits
        if (q1_min - limit_epsilon <= q1_rad_normalized <= q1_max + limit_epsilon) and \
           (q2_min - limit_epsilon <= q2_rad_normalized <= q2_max + limit_epsilon):
            return [q1_rad_normalized, q2_rad_normalized] # Return the first valid solution found

    return None # No solution found within joint limits
