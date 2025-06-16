import math
import numpy as np
from .model import RobotArm

# --- Matrix and Transformation Helpers (using NumPy) ---
def create_translation_matrix_np(dx, dy, dz):
    return np.array([
        [1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]
    ], dtype=np.float64)

def create_rotation_matrix_np(axis_char: str, angle_rad: float):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    if axis_char == 'x': return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=np.float64)
    elif axis_char == 'y': return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=np.float64)
    elif axis_char == 'z': return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    else: raise ValueError(f"Unsupported joint axis: {axis_char}")

def get_joint_frames_numpy(robot_arm: RobotArm, joint_angles: list[float]) -> list[np.ndarray]:
    if len(joint_angles) != robot_arm.num_joints:
        raise ValueError("Number of joint angles must match number of joints.")
    frames = [np.identity(4, dtype=np.float64)]
    current_global_transform = np.identity(4, dtype=np.float64)
    for i in range(robot_arm.num_joints):
        angle_rad = joint_angles[i]
        bone_length = robot_arm.bone_lengths[i]
        axis_char = robot_arm.joint_axes[i].lower()
        rot_matrix_local = create_rotation_matrix_np(axis_char, angle_rad)
        trans_matrix_local = create_translation_matrix_np(bone_length, 0, 0)
        current_global_transform = current_global_transform @ rot_matrix_local @ trans_matrix_local
        frames.append(current_global_transform.copy())
    return frames

def get_joint_positions_numpy(robot_arm: RobotArm, joint_angles: list[float]) -> list[np.ndarray]:
    frames = get_joint_frames_numpy(robot_arm, joint_angles)
    positions = [frame[:3, 3] for frame in frames]
    return positions

def calculate_forward_kinematics(robot_arm: RobotArm, joint_angles: list[float]) -> tuple[float, float, float]:
    all_positions = get_joint_positions_numpy(robot_arm, joint_angles)
    ee_pos_np = all_positions[-1]
    return (ee_pos_np[0], ee_pos_np[1], ee_pos_np[2])

def calculate_jacobian(robot_arm: RobotArm, joint_frames_world: list[np.ndarray]) -> np.ndarray:
    num_joints = robot_arm.num_joints
    jacobian = np.zeros((3, num_joints), dtype=np.float64)
    P_ee = joint_frames_world[num_joints][:3, 3]
    for i in range(num_joints):
        P_i = joint_frames_world[i][:3, 3]
        joint_local_axis_char = robot_arm.joint_axes[i].lower()
        local_rotation_axis_vec = np.array([0.,0.,0.])
        if joint_local_axis_char == 'x': local_rotation_axis_vec = np.array([1.,0.,0.])
        elif joint_local_axis_char == 'y': local_rotation_axis_vec = np.array([0.,1.,0.])
        elif joint_local_axis_char == 'z': local_rotation_axis_vec = np.array([0.,0.,1.])
        else: raise ValueError(f"Invalid joint axis '{joint_local_axis_char}' for joint {i}")
        R_i_world = joint_frames_world[i][:3, :3]
        Z_i_world = R_i_world @ local_rotation_axis_vec
        norm_zi = np.linalg.norm(Z_i_world)
        if norm_zi > 1e-9: Z_i_world /= norm_zi
        jacobian[:, i] = np.cross(Z_i_world, P_ee - P_i)
    return jacobian

# --- IK Solvers ---
def _calculate_ik_2_link_planar_analytical(robot_arm: RobotArm, target_position_np: np.ndarray) -> list[float] | None:
    l1, l2 = robot_arm.bone_lengths[0], robot_arm.bone_lengths[1]
    x, y = target_position_np[0], target_position_np[1]
    if l1 <= 0 or l2 <= 0: return None
    dist_sq = x*x + y*y
    epsilon = 1e-9
    if dist_sq > (l1 + l2)**2 + epsilon or dist_sq < (l1 - l2)**2 - epsilon: return None

    denom_cos_q2 = 2 * l1 * l2
    if abs(denom_cos_q2) < epsilon:
        if dist_sq < epsilon and abs(l1-l2) < epsilon :
             q1_special = -math.pi / 2
             q2_special = -math.pi
             q1_min, q1_max = robot_arm.joint_limits[0]
             q2_min, q2_max = robot_arm.joint_limits[1]
             limit_epsilon = 1e-7
             if (q1_min - limit_epsilon <= q1_special <= q1_max + limit_epsilon) and \
                (q2_min - limit_epsilon <= q2_special <= q2_max + limit_epsilon):
                 fk_pos_check = calculate_forward_kinematics(robot_arm, [q1_special, q2_special])
                 if np.linalg.norm(np.array(fk_pos_check)[:2] - target_position_np[:2]) < 1e-5:
                     return [q1_special, q2_special]
        return None

    cos_q2_val = np.clip((dist_sq - l1*l1 - l2*l2) / denom_cos_q2, -1.0, 1.0)
    q2_options = [math.acos(cos_q2_val), -math.acos(cos_q2_val)]
    limit_epsilon_check = 1e-7

    for q2_rad_attempt in q2_options:
        s2, c2 = math.sin(q2_rad_attempt), math.cos(q2_rad_attempt)
        q1_rad_attempt = math.atan2(y, x) - math.atan2(l2*s2, l1+l2*c2)

        q1_rad_normalized = (q1_rad_attempt + math.pi) % (2 * math.pi) - math.pi
        q2_rad_normalized = (q2_rad_attempt + math.pi) % (2 * math.pi) - math.pi

        q1_min, q1_max = robot_arm.joint_limits[0]; q2_min, q2_max = robot_arm.joint_limits[1]

        if (q1_min-limit_epsilon_check <= q1_rad_normalized <= q1_max+limit_epsilon_check) and \
           (q2_min-limit_epsilon_check <= q2_rad_normalized <= q2_max+limit_epsilon_check):
            fk_pos_check = calculate_forward_kinematics(robot_arm, [q1_rad_normalized, q2_rad_normalized])
            if np.linalg.norm(np.array(fk_pos_check)[:2] - target_position_np[:2]) < 1e-5:
                 return [q1_rad_normalized, q2_rad_normalized]
    return None

def _calculate_ik_ccd(robot_arm: RobotArm, target_position_np: np.ndarray,
                      initial_angles: list[float] | None = None,
                      max_iterations: int = 100, tolerance: float = 1e-4) -> list[float] | None:
    num_joints = robot_arm.num_joints
    current_angles_list = list(initial_angles) if initial_angles and len(initial_angles) == num_joints else [0.0] * num_joints
    current_angles_np = np.array(current_angles_list, dtype=np.float64)

    for j_init in range(num_joints):
        q_min, q_max = robot_arm.joint_limits[j_init]
        current_angles_np[j_init] = np.clip(current_angles_np[j_init], q_min, q_max)

    for iteration in range(max_iterations):
        all_joint_frames = get_joint_frames_numpy(robot_arm, current_angles_np.tolist())
        end_effector_pos = all_joint_frames[-1][:3, 3]
        dist_to_target = np.linalg.norm(end_effector_pos - target_position_np)
        if dist_to_target < tolerance: return current_angles_np.tolist()

        for j_ccd in range(num_joints - 1, -1, -1):
            all_joint_frames_inner = get_joint_frames_numpy(robot_arm, current_angles_np.tolist())
            joint_j_frame_world = all_joint_frames_inner[j_ccd]
            end_effector_pos_inner = all_joint_frames_inner[-1][:3,3]

            joint_j_origin_world = joint_j_frame_world[:3,3]
            vec_joint_to_ee = end_effector_pos_inner - joint_j_origin_world
            vec_joint_to_target = target_position_np - joint_j_origin_world

            local_axis_char = robot_arm.joint_axes[j_ccd]
            joint_axis_local_3d = np.array([1.0 if local_axis_char == 'x' else 0.0,
                                          1.0 if local_axis_char == 'y' else 0.0,
                                          1.0 if local_axis_char == 'z' else 0.0], dtype=np.float64)

            joint_axis_world = joint_j_frame_world[:3,:3] @ joint_axis_local_3d
            norm_jaw = np.linalg.norm(joint_axis_world)
            if norm_jaw < 1e-9: continue
            joint_axis_world /= norm_jaw

            v1 = vec_joint_to_ee - np.dot(vec_joint_to_ee, joint_axis_world) * joint_axis_world
            v2 = vec_joint_to_target - np.dot(vec_joint_to_target, joint_axis_world) * joint_axis_world
            norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm_v1 < 1e-9 or norm_v2 < 1e-9: continue

            v1_unit, v2_unit = v1 / norm_v1, v2 / norm_v2
            cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
            delta_angle = math.acos(cos_angle)

            if abs(delta_angle) < 1e-7: continue

            cross_prod_dir = np.cross(v1_unit, v2_unit)
            if np.dot(joint_axis_world, cross_prod_dir) < 0: delta_angle = -delta_angle

            current_angles_np[j_ccd] += delta_angle
            q_min, q_max = robot_arm.joint_limits[j_ccd]
            current_angles_np[j_ccd] = np.clip(current_angles_np[j_ccd], q_min, q_max)

    final_frames = get_joint_frames_numpy(robot_arm, current_angles_np.tolist())
    final_ee_pos = final_frames[-1][:3,3]
    if np.linalg.norm(final_ee_pos - target_position_np) < tolerance * 1.5: return current_angles_np.tolist()
    return None

def _calculate_ik_dls(robot_arm: RobotArm, target_position_np: np.ndarray,
                      initial_angles: list[float] | None = None,
                      max_iterations: int = 100, tolerance: float = 1e-4,
                      damping_factor: float = 0.1, step_size: float = 0.1) -> list[float] | None:
    num_joints = robot_arm.num_joints
    current_angles = np.array(initial_angles if initial_angles and len(initial_angles) == num_joints else [0.0] * num_joints, dtype=np.float64)

    for j_clamp in range(num_joints):
        q_min, q_max = robot_arm.joint_limits[j_clamp]
        current_angles[j_clamp] = np.clip(current_angles[j_clamp], q_min, q_max)

    for iteration in range(max_iterations):
        joint_frames = get_joint_frames_numpy(robot_arm, list(current_angles))
        current_ee_pos = joint_frames[-1][:3, 3]

        error_vec = target_position_np - current_ee_pos
        error_norm = np.linalg.norm(error_vec)

        if error_norm < tolerance:
            return list(current_angles)

        jacobian = calculate_jacobian(robot_arm, joint_frames)

        J_JT = jacobian @ jacobian.T
        lambda_sq_I = damping_factor**2 * np.identity(J_JT.shape[0])

        try:
            inv_term = np.linalg.inv(J_JT + lambda_sq_I)
        except np.linalg.LinAlgError:
            return None

        delta_q = jacobian.T @ inv_term @ error_vec
        current_angles += step_size * delta_q

        for j_clamp_loop in range(num_joints):
            q_min, q_max = robot_arm.joint_limits[j_clamp_loop]
            current_angles[j_clamp_loop] = np.clip(current_angles[j_clamp_loop], q_min, q_max)

    final_joint_frames = get_joint_frames_numpy(robot_arm, list(current_angles))
    final_ee_pos = final_joint_frames[-1][:3, 3]
    final_error_norm = np.linalg.norm(final_ee_pos - target_position_np)

    if final_error_norm < tolerance * 1.5:
        return list(current_angles)
    return None

def calculate_inverse_kinematics(robot_arm: RobotArm, target_position: tuple[float, float, float],
                                 initial_angles: list[float] | None = None,
                                 max_iterations: int = 100, tolerance: float = 1e-4,
                                 solver: str = "dls",
                                 damping_factor: float = 0.1,
                                 step_size: float = 0.1
                                 ) -> list[float] | None:
    target_position_np = np.array(target_position, dtype=np.float64)

    is_2_link_zz = robot_arm.num_joints == 2 and \
                   robot_arm.joint_axes[0].lower()=='z' and \
                   robot_arm.joint_axes[1].lower()=='z'

    if solver.lower() == "analytical_force" and is_2_link_zz:
        return _calculate_ik_2_link_planar_analytical(robot_arm, target_position_np)
    # If not forcing analytical, and it's a 2-link ZZ, and solver is not explicitly an iterative one, use analytical.
    elif is_2_link_zz and solver.lower() not in ["dls", "ccd", "dls_force", "ccd_force"]:
        return _calculate_ik_2_link_planar_analytical(robot_arm, target_position_np)

    # Explicit iterative solvers or default DLS for general case
    if solver.lower() == "dls" or solver.lower() == "dls_force":
        return _calculate_ik_dls(robot_arm, target_position_np, initial_angles, max_iterations, tolerance, damping_factor, step_size)
    elif solver.lower() == "ccd" or solver.lower() == "ccd_force":
        return _calculate_ik_ccd(robot_arm, target_position_np, initial_angles, max_iterations, tolerance)
    else:
        # Fallback for unrecognized solver string for N-link arm, default to DLS
        return _calculate_ik_dls(robot_arm, target_position_np, initial_angles, max_iterations, tolerance, damping_factor, step_size)
