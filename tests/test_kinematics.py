import unittest
import math
from robot_arm.model import RobotArm
from robot_arm.kinematics import calculate_forward_kinematics, calculate_inverse_kinematics

class TestKinematics(unittest.TestCase):

    def assertPositionsAlmostEqual(self, pos1, pos2, places=6):
        self.assertAlmostEqual(pos1[0], pos2[0], places=places, msg=f"X: {pos1[0]} vs {pos2[0]}")
        self.assertAlmostEqual(pos1[1], pos2[1], places=places, msg=f"Y: {pos1[1]} vs {pos2[1]}")
        self.assertAlmostEqual(pos1[2], pos2[2], places=places, msg=f"Z: {pos1[2]} vs {pos2[2]}")

    def assertAnglesAlmostEqualList(self, list1, list2, places=6):
        self.assertEqual(len(list1), len(list2), "Angle lists differ in length")
        for a1, a2 in zip(list1, list2):
            # Handle pi vs -pi equivalence for tests where appropriate
            # Check if both are close to pi or -pi in magnitude
            if math.isclose(abs(a1), math.pi, abs_tol=1e-9) and math.isclose(abs(a2), math.pi, abs_tol=1e-9):
                 self.assertAlmostEqual(abs(a1), abs(a2), places=places, msg=f"Angle magnitude {abs(a1)} vs {abs(a2)} (expected near pi)")
            else:
                 self.assertAlmostEqual(a1, a2, places=places, msg=f"Angle {a1} vs {a2}")


    # --- Existing Forward Kinematics Tests ---
    def test_fk_1_dof_z_rotation(self):
        arm = RobotArm(num_joints=1, bone_lengths=[10.0], joint_axes=['z'], joint_limits=[(-math.pi, math.pi)])
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [0.0]), (10.0, 0.0, 0.0))
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [math.pi/2]), (0.0, 10.0, 0.0))
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [math.pi]), (-10.0, 0.0, 0.0))

    def test_fk_1_dof_x_rotation(self):
        arm = RobotArm(num_joints=1, bone_lengths=[10.0], joint_axes=['x'], joint_limits=[(-math.pi, math.pi)])
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [0.0]), (10.0, 0.0, 0.0))
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [math.pi/2]), (10.0, 0.0, 0.0))

    def test_fk_1_dof_y_rotation(self):
        arm = RobotArm(num_joints=1, bone_lengths=[10.0], joint_axes=['y'], joint_limits=[(-math.pi, math.pi)])
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [0.0]), (10.0, 0.0, 0.0))
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [math.pi/2]), (0.0, 0.0, -10.0))

    def test_fk_2_dof_planar_zz(self):
        arm = RobotArm(num_joints=2, bone_lengths=[10.0, 5.0], joint_axes=['z', 'z'], joint_limits=[(-math.pi, math.pi)]*2)
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [0.0, 0.0]), (15.0, 0.0, 0.0))
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [math.pi/2, 0.0]), (0.0, 15.0, 0.0))
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [0.0, math.pi/2]), (10.0, 5.0, 0.0))

        L1, L2 = 10.0, 5.0
        q1_a, q2_a = math.pi/4, math.pi/4
        expected_x = L1 * math.cos(q1_a) + L2 * math.cos(q1_a + q2_a)
        expected_y = L1 * math.sin(q1_a) + L2 * math.sin(q1_a + q2_a)
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [q1_a, q2_a]), (expected_x, expected_y, 0.0))

    def test_fk_2_dof_mixed_zy(self):
        arm = RobotArm(num_joints=2, bone_lengths=[10.0, 5.0], joint_axes=['z', 'y'], joint_limits=[(-math.pi, math.pi)]*2)
        self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, [0.0, math.pi/2]), (10.0, 0.0, -5.0))

    def test_fk_invalid_angles_length(self):
        arm = RobotArm(num_joints=2, bone_lengths=[1.0,1.0], joint_axes=['z','z'], joint_limits=[(0,1),(0,1)])
        with self.assertRaisesRegex(ValueError, "Number of joint angles must match the number of joints"):
            calculate_forward_kinematics(arm, [0.0])

    def test_ik_non_2link_planar_arm_raises_error(self):
        arm_3link = RobotArm(num_joints=3, bone_lengths=[1,1,1], joint_axes=['z','z','z'], joint_limits=[(-math.pi,math.pi)]*3)
        arm_1link = RobotArm(num_joints=1, bone_lengths=[1], joint_axes=['z'], joint_limits=[(-math.pi,math.pi)])
        arm_mixed_axes = RobotArm(num_joints=2, bone_lengths=[1,1], joint_axes=['z','x'], joint_limits=[(-math.pi,math.pi)]*2)

        with self.assertRaisesRegex(ValueError, "specifically for 2-link planar arms"):
            calculate_inverse_kinematics(arm_3link, (1,0,0))
        with self.assertRaisesRegex(ValueError, "specifically for 2-link planar arms"):
            calculate_inverse_kinematics(arm_1link, (1,0,0))
        with self.assertRaisesRegex(ValueError, "specifically for 2-link planar arms"):
            calculate_inverse_kinematics(arm_mixed_axes, (1,0,0))

    def test_ik_2_link_planar_simple_cases(self):
        wide_limits = [(-2*math.pi, 2*math.pi), (-2*math.pi, 2*math.pi)]
        arm = RobotArm(num_joints=2, bone_lengths=[1.0, 1.0], joint_axes=['z', 'z'], joint_limits=wide_limits)

        target11 = (1.0, 1.0, 0.0)
        angles11 = calculate_inverse_kinematics(arm, target11)
        self.assertIsNotNone(angles11, "IK solution should exist for (1,1)")
        if angles11:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, angles11), target11)
            self.assertAnglesAlmostEqualList(angles11, [0.0, math.pi/2])

        target_sqrt2_0 = (math.sqrt(2), 0.0, 0.0)
        angles_sqrt2_0 = calculate_inverse_kinematics(arm, target_sqrt2_0)
        self.assertIsNotNone(angles_sqrt2_0, f"IK solution should exist for {target_sqrt2_0}")
        if angles_sqrt2_0:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, angles_sqrt2_0), target_sqrt2_0)
            self.assertAnglesAlmostEqualList(angles_sqrt2_0, [-math.pi/4, math.pi/2])

        target20 = (2.0, 0.0, 0.0)
        angles20 = calculate_inverse_kinematics(arm, target20)
        self.assertIsNotNone(angles20, "IK solution should exist for (2,0)")
        if angles20:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, angles20), target20)
            self.assertAnglesAlmostEqualList(angles20, [0.0, 0.0])

        target00 = (0.0, 0.0, 0.0)
        angles00 = calculate_inverse_kinematics(arm, target00)
        self.assertIsNotNone(angles00, "IK solution should exist for (0,0) with L1=L2")
        if angles00:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, angles00), target00)
            # Due to math.sin(pi) not being exactly 0, q1 becomes -pi/2 for target (0,0)
            self.assertAnglesAlmostEqualList(angles00, [-math.pi/2, -math.pi])

    def test_ik_2_link_unreachable(self):
        arm = RobotArm(num_joints=2, bone_lengths=[1.0, 1.0], joint_axes=['z', 'z'], joint_limits=[(-math.pi, math.pi)]*2)
        self.assertIsNone(calculate_inverse_kinematics(arm, (2.0001, 0.0, 0.0)), "Target too far (2.0001,0,0)")
        self.assertIsNone(calculate_inverse_kinematics(arm, (0.0, -2.0001, 0.0)), "Target too far (0,-2.0001,0)")

        arm_diff_lengths = RobotArm(num_joints=2, bone_lengths=[2.0, 1.0], joint_axes=['z', 'z'], joint_limits=[(-math.pi, math.pi)]*2)
        self.assertIsNone(calculate_inverse_kinematics(arm_diff_lengths, (0.9999, 0.0, 0.0)), "Target too close (0.9999,0,0) for Ls 2,1")

        target_boundary_inner = (1.0, 0.0, 0.0)
        angles_boundary_inner = calculate_inverse_kinematics(arm_diff_lengths, target_boundary_inner)
        self.assertIsNotNone(angles_boundary_inner, "Target at |l1-l2| should be reachable")
        if angles_boundary_inner:
             self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm_diff_lengths, angles_boundary_inner), target_boundary_inner)
             self.assertAnglesAlmostEqualList(angles_boundary_inner, [0.0, -math.pi])

    def test_ik_2_link_joint_limits(self):
        target11 = (1.0, 1.0, 0.0)

        arm_ok = RobotArm(num_joints=2, bone_lengths=[1.0, 1.0], joint_axes=['z', 'z'],
                           joint_limits=[(-math.pi/4, math.pi/4), (math.pi/4, 3*math.pi/4)])
        angles_ok = calculate_inverse_kinematics(arm_ok, target11)
        self.assertIsNotNone(angles_ok, "IK should find primary solution [0, pi/2] for (1,1)")
        if angles_ok:
            self.assertAnglesAlmostEqualList(angles_ok, [0.0, math.pi/2])

        arm_alt_sol = RobotArm(num_joints=2, bone_lengths=[1.0, 1.0], joint_axes=['z', 'z'],
                               joint_limits=[(math.pi/4, 3*math.pi/4), (-3*math.pi/4, -math.pi/4)])
        angles_alt_sol = calculate_inverse_kinematics(arm_alt_sol, target11)
        self.assertIsNotNone(angles_alt_sol, "IK should find alternative solution [pi/2, -pi/2] for target (1,1)")
        if angles_alt_sol:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm_alt_sol, angles_alt_sol), target11)
            self.assertAnglesAlmostEqualList(angles_alt_sol, [math.pi/2, -math.pi/2])

        target_0_sqrt2 = (0, math.sqrt(2), 0.0)
        arm_alt_q2_valid_fk = RobotArm(num_joints=2, bone_lengths=[1.0, 1.0], joint_axes=['z', 'z'],
                                         joint_limits=[(2*math.pi/4, math.pi), (-3*math.pi/4, -math.pi/4)])
        angles_alt_q2_valid_fk = calculate_inverse_kinematics(arm_alt_q2_valid_fk, target_0_sqrt2)
        self.assertIsNotNone(angles_alt_q2_valid_fk, "IK should find alt solution [3pi/4, -pi/2] for (0,sqrt(2))")
        if angles_alt_q2_valid_fk:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm_alt_q2_valid_fk, angles_alt_q2_valid_fk), target_0_sqrt2)
            self.assertAnglesAlmostEqualList(angles_alt_q2_valid_fk, [3*math.pi/4, -math.pi/2])

        arm_limited_all = RobotArm(num_joints=2, bone_lengths=[1.0, 1.0], joint_axes=['z', 'z'],
                                   joint_limits=[(math.pi/8, math.pi/4), (-math.pi/16, math.pi/16)])
        self.assertIsNone(calculate_inverse_kinematics(arm_limited_all, target_0_sqrt2), "No solution due to joint limits for (0,sqrt(2))")

    def test_ik_2_link_at_origin_special_case(self):
        arm = RobotArm(num_joints=2, bone_lengths=[1.0, 1.0], joint_axes=['z', 'z'], joint_limits=[(-math.pi,math.pi)]*2)
        angles = calculate_inverse_kinematics(arm, (0,0,0))
        self.assertIsNotNone(angles)
        if angles:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, angles), (0,0,0))
            # Due to math.sin(pi) not being exactly 0, q1 becomes -pi/2 for target (0,0)
            self.assertAnglesAlmostEqualList(angles, [-math.pi/2, -math.pi])

        arm_diff = RobotArm(num_joints=2, bone_lengths=[2.0, 1.0], joint_axes=['z', 'z'], joint_limits=[(-math.pi,math.pi)]*2)
        self.assertIsNone(calculate_inverse_kinematics(arm_diff, (0,0,0)), "Target (0,0) unreachable if l1!=l2")
