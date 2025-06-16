import unittest
import math
from robot_arm.model import RobotArm

class TestRobotArm(unittest.TestCase):

    def test_valid_initialization(self):
        arm = RobotArm(num_joints=2, bone_lengths=[1.0, 0.5], joint_axes=['z', 'x'], joint_limits=[(-math.pi, math.pi), (0, math.pi/2)])
        self.assertEqual(arm.get_num_joints(), 2)
        self.assertEqual(arm.get_bone_lengths(), [1.0, 0.5])
        self.assertEqual(arm.get_joint_axes(), ['z', 'x'])
        self.assertEqual(arm.get_joint_limits(), [(-math.pi, math.pi), (0, math.pi/2)])

    def test_invalid_num_joints(self):
        with self.assertRaisesRegex(ValueError, "Number of joints must be a positive integer"):
            RobotArm(num_joints=0, bone_lengths=[], joint_axes=[], joint_limits=[])
        with self.assertRaisesRegex(ValueError, "Number of joints must be a positive integer"):
            RobotArm(num_joints=-1, bone_lengths=[], joint_axes=[], joint_limits=[])

    def test_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, "Length of bone_lengths, joint_axes, and joint_limits must match num_joints"):
            RobotArm(num_joints=2, bone_lengths=[1.0], joint_axes=['z','z'], joint_limits=[(0,1),(0,1)])
        with self.assertRaisesRegex(ValueError, "Length of bone_lengths, joint_axes, and joint_limits must match num_joints"):
            RobotArm(num_joints=2, bone_lengths=[1.0, 0.5], joint_axes=['z'], joint_limits=[(0,1),(0,1)])
        with self.assertRaisesRegex(ValueError, "Length of bone_lengths, joint_axes, and joint_limits must match num_joints"):
            RobotArm(num_joints=2, bone_lengths=[1.0, 0.5], joint_axes=['z','z'], joint_limits=[(0,1)])

    def test_invalid_joint_axes(self):
        with self.assertRaisesRegex(ValueError, "Joint axes must be 'x', 'y', or 'z'"):
            RobotArm(num_joints=1, bone_lengths=[1.0], joint_axes=['a'], joint_limits=[(0,1)])

    def test_invalid_joint_limits_format(self):
        with self.assertRaisesRegex(ValueError, r"Joint limits must be tuples of \(min_angle, max_angle\)"): # Raw string for regex
            RobotArm(num_joints=1, bone_lengths=[1.0], joint_axes=['x'], joint_limits=[(0,1,2)]) # Wrong tuple size
        with self.assertRaisesRegex(ValueError, r"Joint limits must be tuples of \(min_angle, max_angle\)"): # Raw string for regex
            RobotArm(num_joints=1, bone_lengths=[1.0], joint_axes=['x'], joint_limits=[[0,1]]) # Not a tuple
        with self.assertRaisesRegex(ValueError, r"Joint limits must be tuples of \(min_angle, max_angle\)"): # Raw string for regex
            RobotArm(num_joints=1, bone_lengths=[1.0], joint_axes=['x'], joint_limits=[(1,0)]) # min > max

    def test_setters_valid(self):
        arm = RobotArm(num_joints=1, bone_lengths=[1.0], joint_axes=['z'], joint_limits=[(0,math.pi)])
        arm.set_bone_lengths([2.0])
        self.assertEqual(arm.get_bone_lengths(), [2.0])
        arm.set_joint_axes(['x'])
        self.assertEqual(arm.get_joint_axes(), ['x'])
        arm.set_joint_limits([(0, math.pi/2)])
        self.assertEqual(arm.get_joint_limits(), [(0, math.pi/2)])

    def test_setters_invalid(self):
        arm = RobotArm(num_joints=2, bone_lengths=[1.0,1.0], joint_axes=['z','z'], joint_limits=[(0,1),(0,1)])
        with self.assertRaisesRegex(ValueError, "Length of bone_lengths must match the number of joints"):
            arm.set_bone_lengths([2.0])
        with self.assertRaisesRegex(ValueError, "Length of joint_axes must match the number of joints"):
            arm.set_joint_axes(['x'])
        with self.assertRaisesRegex(ValueError, "Joint axes must be 'x', 'y', or 'z'"):
            arm.set_joint_axes(['a', 'y'])
        with self.assertRaisesRegex(ValueError, "Length of joint_limits must match the number of joints"):
            arm.set_joint_limits([(0,0.5)])
        with self.assertRaisesRegex(ValueError, r"Joint limits must be tuples of \(min_angle, max_angle\)"): # Raw string for regex
            arm.set_joint_limits([(0,0.5), (1,0)])
