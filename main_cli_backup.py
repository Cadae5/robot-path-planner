import math
import os
import numpy as np # For target conversion if needed, and general use
import matplotlib.pyplot as plt # Moved here as it's essential now

from robot_arm.model import RobotArm
from robot_arm.kinematics import calculate_forward_kinematics, calculate_inverse_kinematics
from gcode_processor.parser import load_gcode
from visualization.viewer import draw_arm


def main():
    print("Robot Arm Control CLI - Using CCD IK Solver for N-Link Arm")
    pi = math.pi

    # 1. Define a 3-link planar robot arm (ZZZ configuration)
    # This arm configuration should work reasonably well with the current CCD.
    robot_config = {
        'num_joints': 3,
        'bone_lengths': [1.0, 0.8, 0.6], # L1, L2, L3
        'joint_axes': ['z', 'z', 'z'],   # All joints rotate around Z-axis
        'joint_limits': [(-pi, pi), (-pi, pi), (-pi, pi)] # Wide open limits
    }
    # Max reach approx 1.0 + 0.8 + 0.6 = 2.4

    # Uncomment to test the ZXZ arm (known to be challenging for current CCD)
    # robot_config = {
    #     'num_joints': 3,
    #     'bone_lengths': [1.0, 1.0, 0.5],
    #     'joint_axes': ['z', 'x', 'z'],
    #     'joint_limits': [(-pi, pi), (-pi/2, pi/2), (-pi, pi)]
    # }


    my_robot = RobotArm(**robot_config)
    print(f"Robot Arm Defined: {my_robot.num_joints} joints")
    print(f"  Lengths: {my_robot.bone_lengths}")
    print(f"  Axes: {my_robot.joint_axes}")
    print(f"  Limits: {[(f'{l[0]:.2f}', f'{l[1]:.2f}') for l in my_robot.joint_limits]}")

    current_max_reach = sum(my_robot.get_bone_lengths())
    print(f"Calculated Max Reach: {current_max_reach:.2f}")

    # --- Visualization Setup ---
    fig = plt.figure(figsize=(9,8)) # Slightly larger figure
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()  # Interactive mode ON
    fig.show()
    print("NOTE: Matplotlib window might be behind other windows or require focus.")
    # --- End Visualization Setup ---

    # 2. Create a sample G-code file with 3D targets for the 3-link arm
    sample_gcode_content = f"""
; Test cases for 3-link ZZZ arm (L_total={current_max_reach:.2f})
G0 X{current_max_reach*0.8:.2f} Y0.0 Z0.0  ; Reachable, straight-ish X
G1 X0.0 Y{current_max_reach*0.7:.2f} Z0.0  ; Reachable, straight-ish Y
G0 X0.5 Y0.5 Z0.0                 ; Reachable, general planar position
G1 X1.0 Y-0.5 Z0.0                ; Reachable, another planar position
; Test some spatial targets (though arm is planar ZZZ, target Z will be ignored by FK for ZZZ)
; For a ZZZ arm, FK will always produce Z=0. CCD will try to match target Z if it can.
; The current get_joint_positions_numpy for ZZZ arm will always have Z=0.
G0 X0.7 Y0.7 Z0.5                 ; Target Z is non-zero
G1 X-0.5 Y-0.5 Z-0.3               ; Target Z is non-zero
; Unreachable
G0 X{current_max_reach*1.2:.2f} Y0.0 Z0.0  ; Unreachable (too far)
"""
    sample_gcode_filepath = "sample_cli_ccd.gcode"
    with open(sample_gcode_filepath, 'w') as f:
        f.write(sample_gcode_content)
    print(f"Sample G-code file created: {sample_gcode_filepath}")

    # 3. Load and parse the G-code
    parsed_commands = load_gcode(sample_gcode_filepath)
    if not parsed_commands:
        print("No G-code commands parsed. Exiting.")
        if os.path.exists(sample_gcode_filepath): os.remove(sample_gcode_filepath)
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.ioff(); plt.show() # Ensure plot interaction ends
        return

    print(f"\n--- Processing {len(parsed_commands)} G-code commands ---")

    prev_angles = [0.0] * my_robot.num_joints

    print("Drawing initial arm position (all zeros)...")
    draw_arm(ax, my_robot, prev_angles, target_position=None, max_reach=current_max_reach)
    ax.set_title('Robot Arm Simulation (Initial Position)')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(1.0)


    # 4. Process commands
    for i, command_data in enumerate(parsed_commands):
        print(f"\nCommand {i+1}: {command_data['command']}")

        target_x = command_data.get('x', 0.0)
        target_y = command_data.get('y', 0.0)
        target_z = command_data.get('z', 0.0)

        current_target_position = (target_x, target_y, target_z)
        print(f"  G-code Target: X={current_target_position[0]:.3f}, Y={current_target_position[1]:.3f}, Z={current_target_position[2]:.3f}")

        calculated_angles = None
        try:
            solver_type = 'CCD'
            if my_robot.num_joints == 2 and \
               my_robot.joint_axes[0].lower() == 'z' and \
               my_robot.joint_axes[1].lower() == 'z':
                solver_type = 'Analytical (2-Link Planar)'

            print(f"  Attempting IK (using {solver_type}) with initial_angles: {[f'{a:.2f}' for a in prev_angles]}")
            calculated_angles = calculate_inverse_kinematics(my_robot, current_target_position,
                                                             initial_angles=prev_angles,
                                                             max_iterations=500,
                                                             tolerance=1e-3)

            if calculated_angles:
                q_degrees = [math.degrees(q) for q in calculated_angles]
                print(f"  IK Solution: Q_rad={[f'{q:.3f}' for q in calculated_angles]}, Q_deg={[f'{qd:.1f}' for qd in q_degrees]}")

                fk_position = calculate_forward_kinematics(my_robot, calculated_angles)
                print(f"  FK Verification: X={fk_position[0]:.3f}, Y={fk_position[1]:.3f}, Z={fk_position[2]:.3f}")

                dx = fk_position[0] - current_target_position[0]
                dy = fk_position[1] - current_target_position[1]
                dz = fk_position[2] - current_target_position[2]
                dist_error = math.sqrt(dx*dx + dy*dy + dz*dz)
                print(f"  Error (FK vs Target): {dist_error:.6f}")
                prev_angles = calculated_angles
            else:
                print("  IK Solution: Target is unreachable or no solution found by solver within parameters.")

        except ValueError as ve:
            print(f"  IK Error: {ve}")
        except Exception as e:
            print(f"  An unexpected error occurred during IK/FK: {e}")

        angles_to_draw = calculated_angles if calculated_angles else prev_angles
        title = 'Robot Arm Simulation'
        if not calculated_angles and current_target_position:
            title += ' (Target Unreachable or No IK Sol)'

        draw_arm(ax, my_robot, angles_to_draw, target_position=current_target_position, max_reach=current_max_reach)
        ax.set_title(title)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1.0)

    # 5. Clean up
    if os.path.exists(sample_gcode_filepath):
        os.remove(sample_gcode_filepath)
        print(f"\nCleaned up sample G-code file: {sample_gcode_filepath}")

    if 'fig' in locals() and plt.fignum_exists(fig.number):
        plt.ioff()
        print("\nFinished processing all G-code commands.")
        print("Final plot displayed. Close Matplotlib window to exit.")
        plt.show()

if __name__ == '__main__':
    main()
