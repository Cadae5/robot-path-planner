# This file makes Python treat the 'robot_arm' directory as a package.
from .model import RobotArm
from .kinematics import calculate_forward_kinematics, calculate_inverse_kinematics
