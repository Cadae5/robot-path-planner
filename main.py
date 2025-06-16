import math
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Advise Matplotlib to use Tkinter backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import time

from robot_arm.model import RobotArm
from robot_arm.kinematics import calculate_forward_kinematics, calculate_inverse_kinematics
from gcode_processor.parser import load_gcode
from visualization.viewer import draw_arm

# --- Global variables ---
robot_instance: RobotArm | None = None
gcode_filepath_var: tk.StringVar | None = None
output_text_widget: scrolledtext.ScrolledText | None = None

fig_plt: plt.Figure | None = None
ax_plt: plt.Axes | None = None
canvas_tk: FigureCanvasTkAgg | None = None
prev_angles_sim: list[float] = []

DEFAULT_NUM_JOINTS = 3
DEFAULT_BONE_LENGTHS = "1.0, 0.8, 0.6"
DEFAULT_JOINT_AXES = "z, z, z"
DEFAULT_JOINT_LIMITS_MIN = "-3.14, -3.14, -3.14"
DEFAULT_JOINT_LIMITS_MAX = "3.14, 3.14, 3.14"

def log_message(message: str):
    if output_text_widget and output_text_widget.winfo_exists():
        output_text_widget.insert(tk.END, message + "\n")
        output_text_widget.see(tk.END)
        output_text_widget.update_idletasks()
    else:
        print(message)

def validate_and_create_robot(num_joints_str, bone_lengths_str, joint_axes_str,
                              joint_limits_min_str, joint_limits_max_str):
    global robot_instance, prev_angles_sim, fig_plt, ax_plt, canvas_tk # Added fig/ax/canvas for redraw
    log_message("Attempting to define robot...")
    try:
        num_joints = int(num_joints_str)
        if num_joints <= 0: raise ValueError("Number of joints must be positive.")
        bone_lengths = [float(x.strip()) for x in bone_lengths_str.split(',') if x.strip()]
        joint_axes = [x.strip().lower() for x in joint_axes_str.split(',') if x.strip()]
        min_limits_str_list = [x.strip() for x in joint_limits_min_str.split(',') if x.strip()]
        max_limits_str_list = [x.strip() for x in joint_limits_max_str.split(',') if x.strip()]

        if not (len(bone_lengths) == num_joints and len(joint_axes) == num_joints and \
                len(min_limits_str_list) == num_joints and len(max_limits_str_list) == num_joints):
            raise ValueError(f"Mismatch in lengths of inputs for {num_joints} joints.")

        joint_limits = []
        for i in range(num_joints):
            min_val, max_val = float(min_limits_str_list[i]), float(max_limits_str_list[i])
            joint_limits.append((min_val, max_val))

        robot_instance = RobotArm(num_joints, bone_lengths, joint_axes, joint_limits)
        prev_angles_sim = [0.0] * robot_instance.num_joints
        log_message("Robot defined successfully!")
        log_message(f"  Joints: {robot_instance.num_joints}, Lengths: {robot_instance.bone_lengths}")
        log_message(f"  Axes: {robot_instance.joint_axes}, Limits: {robot_instance.joint_limits}")

        # Draw initial state on embedded canvas if it exists
        if robot_instance and canvas_tk and ax_plt:
            current_max_reach = sum(robot_instance.get_bone_lengths()) if robot_instance.bone_lengths else 2.0
            draw_arm(ax_plt, robot_instance, prev_angles_sim, target_position=None, max_reach=current_max_reach)
            ax_plt.set_title('Robot Arm (Initial Position)')
            canvas_tk.draw()
        return True
    except ValueError as ve:
        log_message(f"Error defining robot: {ve}")
    except Exception as e:
        log_message(f"Unexpected error defining robot: {e}")
    robot_instance = None
    prev_angles_sim = []
    return False

def load_gcode_file_dialog():
    if gcode_filepath_var is None: log_message("Error: G-code filepath var not ready."); return
    filepath = filedialog.askopenfilename(
        title="Select G-code File",
        filetypes=(("G-code files", "*.gcode *.gc *.gco *.nc"), ("All files", "*.*"))
    )
    if filepath: gcode_filepath_var.set(filepath); log_message(f"G-code file: {filepath}")
    else: log_message("No G-code file selected.")

def run_simulation():
    global robot_instance, fig_plt, ax_plt, canvas_tk, prev_angles_sim

    if not output_text_widget : print("Error: Output text widget not available."); return # Should not happen
    if robot_instance is None: log_message("Error: Robot not defined."); return
    filepath = gcode_filepath_var.get() if gcode_filepath_var else None
    if not filepath or not os.path.exists(filepath):
        log_message(f"Error: G-code file missing: '{filepath}'"); return

    if fig_plt is None or ax_plt is None or canvas_tk is None: # Should be set up by main_gui
        log_message("Error: Matplotlib canvas not initialized properly."); return

    log_message(f"Starting simulation: {robot_instance.num_joints}j, G-code: {filepath}")
    # ax_plt.clear() # draw_arm should handle clearing

    parsed_commands = load_gcode(filepath)
    if not parsed_commands: log_message("No G-code commands parsed."); return

    log_message(f"--- Processing {len(parsed_commands)} G-code commands ---")
    if not prev_angles_sim or len(prev_angles_sim) != robot_instance.num_joints:
        prev_angles_sim = [0.0] * robot_instance.num_joints

    current_max_reach = sum(robot_instance.get_bone_lengths()) if robot_instance.bone_lengths else 2.0

    # Initial draw for this simulation run (current state before G-code)
    draw_arm(ax_plt, robot_instance, prev_angles_sim, None, current_max_reach)
    ax_plt.set_title('Robot Arm (Start Simulation)')
    canvas_tk.draw()
    if output_text_widget.winfo_exists(): output_text_widget.winfo_toplevel().update()
    time.sleep(0.5)


    for i, command_data in enumerate(parsed_commands):
        log_message(f"\nCommand {i+1}: {command_data['command']}")
        if output_text_widget.winfo_exists(): output_text_widget.winfo_toplevel().update_idletasks()

        target_x = command_data.get('x',0.0); target_y = command_data.get('y',0.0); target_z = command_data.get('z',0.0)
        current_target_position = (target_x, target_y, target_z)
        log_message(f"  G-code Target: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}")

        calculated_angles = None
        try:
            log_message(f"  Attempting IK with initial_angles: {[f'{a:.2f}' for a in prev_angles_sim]}")
            calculated_angles = calculate_inverse_kinematics(
                robot_instance, current_target_position, initial_angles=prev_angles_sim,
                max_iterations=500, tolerance=1e-3, solver="dls"
            )
            if calculated_angles:
                log_message(f"  IK Sol: Q_rad={[f'{q:.3f}' for q in calculated_angles]}")
                fk_pos = calculate_forward_kinematics(robot_instance, calculated_angles)
                log_message(f"  FK Verify: X={fk_pos[0]:.3f}, Y={fk_pos[1]:.3f}, Z={fk_pos[2]:.3f}")
                err = np.linalg.norm(np.array(fk_pos) - np.array(current_target_position))
                log_message(f"  Error: {err:.6f}")
                prev_angles_sim = calculated_angles
            else:
                log_message("  IK Solution: Target unreachable or no solution found.")
        except Exception as e:
            log_message(f"  Error during IK/FK: {e}")

        angles_to_draw = calculated_angles if calculated_angles else prev_angles_sim
        title = 'Robot Arm Simulation'
        if not calculated_angles: title += ' (Target Unreachable / No IK Sol)'

        draw_arm(ax_plt, robot_instance, angles_to_draw, current_target_position, current_max_reach)
        ax_plt.set_title(title)
        canvas_tk.draw()

        if output_text_widget.winfo_exists(): output_text_widget.winfo_toplevel().update()
        time.sleep(0.1) # Reduced sleep, canvas.draw should be somewhat blocking

    log_message("\nSimulation finished.")
    # Plot is embedded, so it stays until Tk window is closed. No separate plt.show() needed.

def setup_matplotlib_canvas(parent_frame):
    global fig_plt, ax_plt, canvas_tk
    # Use plt.Figure for embedding, not plt.figure()
    fig_plt = plt.Figure(figsize=(6, 5), dpi=100)
    ax_plt = fig_plt.add_subplot(111, projection='3d')
    ax_plt.set_xlabel("X")
    ax_plt.set_ylabel("Y")
    ax_plt.set_zlabel("Z")
    ax_plt.set_title("3D Robot View") # Initial title

    canvas_tk = FigureCanvasTkAgg(fig_plt, master=parent_frame)
    canvas_widget = canvas_tk.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas_tk.draw()
    log_message("Matplotlib canvas initialized and embedded.")


def main_gui():
    global gcode_filepath_var, output_text_widget, prev_angles_sim

    root = tk.Tk()
    root.title("Robot Arm Control Interface")
    root.geometry("950x750")

    gcode_filepath_var = tk.StringVar()
    # prev_angles_sim initialization moved to validate_and_create_robot or before simulation run

    main_paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    main_paned_window.pack(fill=tk.BOTH, expand=True)

    left_pane = ttk.Frame(main_paned_window, width=380, relief=tk.RIDGE, borderwidth=2)
    main_paned_window.add(left_pane, weight=1)

    config_frame = ttk.LabelFrame(left_pane, text="Robot Configuration")
    config_frame.pack(padx=10, pady=10, fill=tk.X, side=tk.TOP)

    ttk.Label(config_frame, text="Num Joints:").grid(row=0,column=0,sticky=tk.W,padx=5,pady=2)
    num_joints_var = tk.StringVar(value=str(DEFAULT_NUM_JOINTS))
    ttk.Entry(config_frame,textvariable=num_joints_var,width=30).grid(row=0,column=1,sticky=tk.EW,padx=5,pady=2)
    ttk.Label(config_frame, text="Bone Lengths (csv):").grid(row=1,column=0,sticky=tk.W,padx=5,pady=2)
    bone_lengths_var = tk.StringVar(value=DEFAULT_BONE_LENGTHS)
    ttk.Entry(config_frame,textvariable=bone_lengths_var,width=30).grid(row=1,column=1,sticky=tk.EW,padx=5,pady=2)
    ttk.Label(config_frame, text="Joint Axes (csv):").grid(row=2,column=0,sticky=tk.W,padx=5,pady=2)
    joint_axes_var = tk.StringVar(value=DEFAULT_JOINT_AXES)
    ttk.Entry(config_frame,textvariable=joint_axes_var,width=30).grid(row=2,column=1,sticky=tk.EW,padx=5,pady=2)
    ttk.Label(config_frame, text="Min Limits (rad,csv):").grid(row=3,column=0,sticky=tk.W,padx=5,pady=2)
    joint_limits_min_var = tk.StringVar(value=DEFAULT_JOINT_LIMITS_MIN)
    ttk.Entry(config_frame,textvariable=joint_limits_min_var,width=30).grid(row=3,column=1,sticky=tk.EW,padx=5,pady=2)
    ttk.Label(config_frame, text="Max Limits (rad,csv):").grid(row=4,column=0,sticky=tk.W,padx=5,pady=2)
    joint_limits_max_var = tk.StringVar(value=DEFAULT_JOINT_LIMITS_MAX)
    ttk.Entry(config_frame,textvariable=joint_limits_max_var,width=30).grid(row=4,column=1,sticky=tk.EW,padx=5,pady=2)
    config_frame.columnconfigure(1, weight=1)
    define_robot_button = ttk.Button(config_frame, text="Define/Update Robot",
        command=lambda: validate_and_create_robot(
            num_joints_var.get(), bone_lengths_var.get(), joint_axes_var.get(),
            joint_limits_min_var.get(), joint_limits_max_var.get()
        ))
    define_robot_button.grid(row=5, column=0, columnspan=2, pady=10)

    gcode_frame = ttk.LabelFrame(left_pane, text="G-code & Simulation")
    gcode_frame.pack(padx=10, pady=10, fill=tk.X, side=tk.TOP) # Changed side
    load_gcode_button = ttk.Button(gcode_frame, text="Load G-code File", command=load_gcode_file_dialog)
    load_gcode_button.pack(pady=5, fill=tk.X)
    gcode_path_display = ttk.Label(gcode_frame, textvariable=gcode_filepath_var, wraplength=300)
    gcode_path_display.pack(pady=2, fill=tk.X); gcode_filepath_var.set("No G-code file selected.")
    run_sim_button = ttk.Button(gcode_frame, text="Run Simulation", command=run_simulation)
    run_sim_button.pack(pady=10, fill=tk.X)

    output_frame = ttk.LabelFrame(left_pane, text="Log / Output")
    output_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True, side=tk.TOP)
    output_text_widget = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10)
    output_text_widget.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    right_pane = ttk.LabelFrame(main_paned_window, text="3D Visualization")
    main_paned_window.add(right_pane, weight=2)
    setup_matplotlib_canvas(right_pane)

    validate_and_create_robot( # Call after output_text_widget and canvas are defined
        num_joints_var.get(), bone_lengths_var.get(), joint_axes_var.get(),
        joint_limits_min_var.get(), joint_limits_max_var.get()
    )
    # Initial draw of default robot is now handled within validate_and_create_robot

    root.mainloop()

if __name__ == '__main__':
    try: import tkinter
    except ImportError: print("tkinter not found."); exit(1)
    try: import matplotlib
    except ImportError: print("matplotlib not found."); exit(1)
    try: import numpy
    except ImportError: print("numpy not found."); exit(1)
    main_gui()
