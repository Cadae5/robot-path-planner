import math
import os

from robot_arm.model import RobotArm
from robot_arm.kinematics import calculate_forward_kinematics, calculate_inverse_kinematics
from gcode_processor.parser import load_gcode

def main():
    print("Robot Arm Control CLI - Using 2D IK Solver")

    # 1. Define a 2-link planar robot arm (compatible with the current IK solver)
    pi = math.pi
    # l1=1.0, l2=0.7. Max reach = 1.7. Min reach = 0.3
    robot_config = {
        'num_joints': 2,
        'bone_lengths': [1.0, 0.7],
        'joint_axes': ['z', 'z'], # Both must be 'z' for the 2D IK solver
        'joint_limits': [(-pi, pi), (-pi, pi)] # Wide open limits for now
    }
    my_robot = RobotArm(**robot_config)
    print(f"Robot Arm Defined: {my_robot.num_joints} joints, lengths {my_robot.bone_lengths}, axes {my_robot.joint_axes}")
    print(f"Max reach: {sum(my_robot.bone_lengths):.3f}, Min reach: {abs(my_robot.bone_lengths[0] - my_robot.bone_lengths[1]):.3f}")


    # 2. Create a sample G-code file with various targets
    sample_gcode_content = """
; Test cases for 2-link planar arm (l1=1.0, l2=0.7)
G0 X1.5 Y0.0 Z0.1  ; Reachable, straight
G1 X0.0 Y1.5 Z0.1  ; Reachable, 90 deg for q1 if q2=0
G0 X0.5 Y0.5 Z0.1  ; Reachable, general position
G1 X1.0 Y1.0 Z0.1  ; Reachable, may require specific elbow
G0 X2.0 Y0.0 Z0.1  ; Unreachable (too far, max reach 1.7)
G1 X0.1 Y0.0 Z0.1  ; Unreachable (too close, min reach 0.3)
G0 X0.8 Y0.0 Z0.1  ; Reachable, q2 near 0
G0 X0.3 Y0.0 Z0.1  ; Reachable, q2 near pi (folded back)
"""
    sample_gcode_filepath = "sample_cli_2d_ik.gcode"
    with open(sample_gcode_filepath, 'w') as f:
        f.write(sample_gcode_content)
    print(f"Sample G-code file created: {sample_gcode_filepath}")

    # 3. Load and parse the G-code
    parsed_commands = load_gcode(sample_gcode_filepath)
    if not parsed_commands:
        print("No G-code commands parsed. Exiting.")
        if os.path.exists(sample_gcode_filepath):
            os.remove(sample_gcode_filepath)
        return

    print(f"\n--- Processing {len(parsed_commands)} G-code commands ---")

    # 4. Process commands
    for i, command_data in enumerate(parsed_commands):
        print(f"\nCommand {i+1}: {command_data['command']}")

        target_x = command_data.get('x', 0.0)
        target_y = command_data.get('y', 0.0)
        target_z = command_data.get('z', 0.0)

        current_target_position = (target_x, target_y, target_z)
        print(f"  G-code Target: X={current_target_position[0]:.3f}, Y={current_target_position[1]:.3f}, Z={current_target_position[2]:.3f}")

        try:
            print(f"  Attempting Inverse Kinematics for target...")
            calculated_angles = calculate_inverse_kinematics(my_robot, current_target_position)

            if calculated_angles:
                q1_deg = math.degrees(calculated_angles[0])
                q2_deg = math.degrees(calculated_angles[1])
                print(f"  IK Solution: q1={calculated_angles[0]:.3f} rad ({q1_deg:.1f} deg), q2={calculated_angles[1]:.3f} rad ({q2_deg:.1f} deg)")

                fk_position = calculate_forward_kinematics(my_robot, calculated_angles)
                print(f"  FK Verification: X={fk_position[0]:.3f}, Y={fk_position[1]:.3f}, Z={fk_position[2]:.3f}")

                dx = fk_position[0] - current_target_position[0]
                dy = fk_position[1] - current_target_position[1]
                dist_error = math.sqrt(dx*dx + dy*dy)
                print(f"  Error (FK vs Target): {dist_error:.6f}")

            else:
                print("  IK Solution: Target is unreachable or no solution within joint limits.")

        except ValueError as ve:
            print(f"  IK Error: {ve}")
        except Exception as e:
            print(f"  An unexpected error occurred during IK/FK: {e}")

    # 5. Clean up
    if os.path.exists(sample_gcode_filepath):
        os.remove(sample_gcode_filepath)
        print(f"\nCleaned up sample G-code file: {sample_gcode_filepath}")

if __name__ == '__main__':
    main()
