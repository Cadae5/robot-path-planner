class RobotArm:
    def __init__(self, num_joints: int, bone_lengths: list[float], joint_axes: list[str], joint_limits: list[tuple[float, float]]):
        if not isinstance(num_joints, int) or num_joints <= 0:
            raise ValueError("Number of joints must be a positive integer.")

        if not (len(bone_lengths) == num_joints and \
                len(joint_axes) == num_joints and \
                len(joint_limits) == num_joints):
            raise ValueError("Length of bone_lengths, joint_axes, and joint_limits must match num_joints.")

        if not all(isinstance(axis, str) and axis.lower() in ['x', 'y', 'z'] for axis in joint_axes):
            raise ValueError("Joint axes must be 'x', 'y', or 'z'.")

        if not all(isinstance(limit, tuple) and len(limit) == 2 and limit[0] <= limit[1] for limit in joint_limits):
            raise ValueError("Joint limits must be tuples of (min_angle, max_angle) with min_angle <= max_angle.")

        self.num_joints = num_joints
        self.bone_lengths = bone_lengths
        self.joint_axes = [axis.lower() for axis in joint_axes]
        self.joint_limits = joint_limits

    def get_num_joints(self) -> int:
        return self.num_joints

    def get_bone_lengths(self) -> list[float]:
        return self.bone_lengths

    def get_joint_axes(self) -> list[str]:
        return self.joint_axes

    def get_joint_limits(self) -> list[tuple[float, float]]:
        return self.joint_limits

    def set_bone_lengths(self, bone_lengths: list[float]):
        if len(bone_lengths) != self.num_joints:
            raise ValueError(f"Length of bone_lengths must match the number of joints ({self.num_joints}).")
        self.bone_lengths = bone_lengths

    def set_joint_axes(self, joint_axes: list[str]):
        if len(joint_axes) != self.num_joints:
            raise ValueError(f"Length of joint_axes must match the number of joints ({self.num_joints}).")
        if not all(isinstance(axis, str) and axis.lower() in ['x', 'y', 'z'] for axis in joint_axes):
            raise ValueError("Joint axes must be 'x', 'y', or 'z'.")
        self.joint_axes = [axis.lower() for axis in joint_axes]

    def set_joint_limits(self, joint_limits: list[tuple[float, float]]):
        if len(joint_limits) != self.num_joints:
            raise ValueError(f"Length of joint_limits must match the number of joints ({self.num_joints}).")
        if not all(isinstance(limit, tuple) and len(limit) == 2 and limit[0] <= limit[1] for limit in joint_limits):
            raise ValueError("Joint limits must be tuples of (min_angle, max_angle) with min_angle <= max_angle.")
        self.joint_limits = joint_limits

    # It's generally not advisable to have a setter for num_joints without careful consideration
    # of how to handle existing dependent attributes (bone_lengths, joint_axes, joint_limits).
    # For now, num_joints is set at instantiation. If it needs to be mutable,
    # a more complex logic for resetting/adjusting other parameters would be needed.
    # def set_num_joints(self, num_joints: int):
    #     if not isinstance(num_joints, int) or num_joints <= 0:
    #         raise ValueError("Number of joints must be a positive integer.")
    #     # Potentially clear or require resetting of other parameters here
    #     self.num_joints = num_joints
