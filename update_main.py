import sys

main_py_path = "main.py"

with open(main_py_path, "r") as f:
    lines = f.readlines()

# Add imports
import_insertion_point = -1
for i, line in enumerate(lines):
    if line.startswith("from gcode_processor.parser import load_gcode"):
        import_insertion_point = i + 1
        break
if import_insertion_point != -1:
    lines.insert(import_insertion_point, "from visualization.viewer import draw_arm\n")
    lines.insert(import_insertion_point, "import matplotlib.pyplot as plt\n")
else:
    # Fallback: add to top (might mess with existing imports if not careful)
    lines.insert(0, "from visualization.viewer import draw_arm\n")
    lines.insert(0, "import matplotlib.pyplot as plt\n")


# Add figure setup in main()
main_func_start = -1
fig_setup_done = False
for i, line in enumerate(lines):
    if "def main():" in line:
        main_func_start = i
    if main_func_start != -1 and "Robot Arm Defined:" in line and not fig_setup_done:
        insert_idx = i + 1
        # Indentation should match the print statement's indentation level
        indentation = ""
        for char_idx, char in enumerate(lines[i]):
            if char.isspace():
                indentation += char
            else:
                # Check if the previous line (the print statement) started with this indent.
                # This is a heuristic. A proper AST parser would be better.
                if lines[i].startswith(indentation):
                    break
                else: # Fallback if line doesn't start with detected indent (e.g. wrapped lines)
                    indentation = "    " # Default to 4 spaces
                    break

        fig_setup_code = [
            indentation + "# --- Visualization Setup ---\n",
            indentation + "fig = plt.figure(figsize=(8,8))\n",
            indentation + "ax = fig.add_subplot(111, projection='3d')\n",
            indentation + "plt.ion()  # Interactive mode ON\n",
            indentation + "fig.show()\n",
            indentation + "print(\"NOTE: Matplotlib window might be behind other windows or require focus.\")\n",
            indentation + "# --- End Visualization Setup ---\n\n"
        ]
        lines[insert_idx:insert_idx] = fig_setup_code
        fig_setup_done = True
        # We need to initialize prev_angles here as well, inside main() scope
        lines.insert(insert_idx + len(fig_setup_code), indentation + "prev_angles = [0.0] * my_robot.num_joints # Initialize prev_angles\n")
        break


# Add draw_arm call and plot updates in the loop
loop_processing_idx = -1
draw_call_done = False
for i, line in enumerate(lines):
    if "--- Processing" in line and "G-code commands ---" in line: # Look for the start of processing
        loop_processing_idx = i

    # Find a good place to insert draw_arm: after FK verification or if IK solution is None
    if loop_processing_idx != -1 and \
       ("print(f\"  Error (FK vs Target):" in line or \
        "print(\"  IK Solution: Target is unreachable" in line) and \
       not draw_call_done:

        insert_idx = i + 1
        indentation = "" # Recalculate indent based on current line
        for char_idx, char in enumerate(lines[i]):
            if char.isspace():
                indentation += char
            else:
                if lines[i].startswith(indentation):
                    break
                else:
                    indentation = "                " # Deeper indent for inside loop and if
                    break

        draw_code = [
            indentation + "# Draw the arm\n",
            indentation + "current_draw_angles = calculated_angles if calculated_angles else prev_angles\n",
            indentation + "if not current_draw_angles: current_draw_angles = [0.0] * my_robot.num_joints\n", # Fallback if prev_angles also not set
            indentation + "current_max_reach = sum(my_robot.get_bone_lengths()) if my_robot else 2.0\n",
            indentation + "draw_title = 'Robot Arm Simulation'\n",
            indentation + "if not calculated_angles: draw_title += ' (Target Unreachable/No IK Sol.)'\n",
            indentation + "draw_arm(ax, my_robot, current_draw_angles, target_position=current_target_position, max_reach=current_max_reach)\n",
            indentation + "ax.set_title(draw_title)\n",
            indentation + "fig.canvas.draw()\n",
            indentation + "fig.canvas.flush_events()\n",
            indentation + "plt.pause(0.5) # Pause to allow plot to update\n",
            indentation + "# Store current angles for next iteration if this one was successful\n",
            indentation + "if calculated_angles: prev_angles = calculated_angles\n"
        ]
        lines[insert_idx:insert_idx] = draw_code
        draw_call_done = True
        break


# Add plt.ioff() and plt.show() at the end of main()
main_func_end = len(lines)
end_of_main_found = False
for i in range(len(lines) -1, -1, -1):
    if "if __name__ == '__main__':" in lines[i]:
        main_func_end = i
        end_of_main_found = True
        break

# Determine indentation for the final plot code based on a line likely inside main()
final_indent = "    " # Default
if fig_setup_done: # If we found where fig setup was, use that indent
    # Find the fig_setup_code line's indent
    for line in lines:
        if "fig = plt.figure(figsize=(8,8))" in line:
            current_indent = ""
            for char in line:
                if char.isspace(): current_indent+=char
                else: break
            final_indent = current_indent
            break


final_plot_code = [
    final_indent + "if 'fig' in locals(): # Only if figure was created\n",
    final_indent + "    plt.ioff() # Interactive mode OFF\n",
    final_indent + "    print(\"Final plot displayed. Close Matplotlib window to exit main program.\")\n",
    final_indent + "    plt.show() # Keep plot open until manually closed\n"
]

if end_of_main_found:
    lines[main_func_end:main_func_end] = final_plot_code
else: # Append if no if __name__ block found (e.g. if main is not wrapped)
    lines.extend(final_plot_code)


with open(main_py_path, "w") as f:
    f.writelines(lines)

print(f"Updated {main_py_path} with Matplotlib visualization calls.")
