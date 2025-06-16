import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d'
import math
import numpy as np

from robot_arm.model import RobotArm
from robot_arm.kinematics import get_joint_positions_numpy # Use the NumPy based one


def draw_arm(ax: plt.Axes, robot_arm_model: RobotArm, joint_angles: list[float],
             target_position: tuple[float,float,float] | None = None,
             max_reach: float = 2.0):
    """
    Draws the robot arm in a Matplotlib 3D subplot.
    Uses get_joint_positions_numpy from kinematics module.
    """
    ax.clear()

    try:
        joint_positions_np = get_joint_positions_numpy(robot_arm_model, joint_angles)
        joint_positions = [tuple(pos_np) for pos_np in joint_positions_np]

    except ValueError as e:
        print(f"Error getting joint positions for drawing: {e}")
        ax.text2D(0.05, 0.95, "Error in arm configuration for drawing.", transform=ax.transAxes, color='red')
        return

    x_coords = [p[0] for p in joint_positions]
    y_coords = [p[1] for p in joint_positions]
    z_coords = [p[2] for p in joint_positions]

    ax.plot(x_coords, y_coords, z_coords, 'o-', color='blue', linewidth=3, markersize=8, label='Arm')
    ax.scatter(x_coords[0], y_coords[0], z_coords[0], s=100, c='black', marker='s', label='Base') # Base is joint_positions[0]
    if len(x_coords) > 1: # Ensure there's more than just the base
      ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], s=100, c='red', marker='x', label='End Effector') # EE is the last point

    if target_position:
        ax.scatter(target_position[0], target_position[1], target_position[2],
                   s=100, c='green', marker='*', label='Target')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Dynamic plot limits based on actual coordinates and max_reach
    all_x_coords = np.array(x_coords + ([target_position[0]] if target_position else []))
    all_y_coords = np.array(y_coords + ([target_position[1]] if target_position else []))
    all_z_coords = np.array(z_coords + ([target_position[2]] if target_position else []))

    if all_x_coords.size == 0: all_x_coords = np.array([-0.1, 0.1]) # Default if no points
    if all_y_coords.size == 0: all_y_coords = np.array([-0.1, 0.1])
    if all_z_coords.size == 0: all_z_coords = np.array([-0.1, 0.1])

    min_x, max_x = np.min(all_x_coords), np.max(all_x_coords)
    min_y, max_y = np.min(all_y_coords), np.max(all_y_coords)
    min_z, max_z = np.min(all_z_coords), np.max(all_z_coords)

    center_x, center_y, center_z = (min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2

    # Determine the largest span needed, considering max_reach as a fallback
    span_x = max(max_x - min_x, 0.2) # Min span to avoid issues with single points
    span_y = max(max_y - min_y, 0.2)
    span_z = max(max_z - min_z, 0.2)

    # Use max_reach to ensure the plot covers potential movement area
    # Fallback to a scaled max_reach if current coordinates are too compact
    plot_half_span = max(span_x, span_y, span_z, max_reach * 0.5) / 2.0 * 1.2 # 20% padding
    plot_half_span = max(plot_half_span, 0.5) # Ensure a minimum view size

    ax.set_xlim(center_x - plot_half_span, center_x + plot_half_span)
    ax.set_ylim(center_y - plot_half_span, center_y + plot_half_span)
    ax.set_zlim(center_z - plot_half_span, center_z + plot_half_span)

    ax.set_title('Robot Arm Simulation')
    ax.legend()
    ax.grid(True)
