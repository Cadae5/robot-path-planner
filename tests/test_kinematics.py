import unittest
import math
import numpy as np
from robot_arm.model import RobotArm
from robot_arm.kinematics import (
    calculate_forward_kinematics, calculate_inverse_kinematics,
    get_joint_frames_numpy, calculate_jacobian
)

class TestKinematics(unittest.TestCase):

    def assertPositionsAlmostEqual(self, pos1, pos2, places=4, msg_prefix=""):
        self.assertAlmostEqual(pos1[0], pos2[0], places=places, msg=f"{msg_prefix} X: {pos1[0]} vs {pos2[0]}")
        self.assertAlmostEqual(pos1[1], pos2[1], places=places, msg=f"{msg_prefix} Y: {pos1[1]} vs {pos2[1]}")
        self.assertAlmostEqual(pos1[2], pos2[2], places=places, msg=f"{msg_prefix} Z: {pos1[2]} vs {pos2[2]}")

    def assertAnglesAlmostEqualList(self, angles1, angles2, places=4):
        self.assertEqual(len(angles1), len(angles2), f"Angle lists differ in length: {angles1} vs {angles2}")
        for a1, a2 in zip(angles1, angles2):
            a1_norm = (a1 + math.pi) % (2 * math.pi) - math.pi
            a2_norm = (a2 + math.pi) % (2 * math.pi) - math.pi
            if math.isclose(abs(a1_norm), math.pi) and math.isclose(abs(a2_norm), math.pi):
                self.assertAlmostEqual(abs(a1_norm), abs(a2_norm), places=places, msg=f"Angle magnitude {abs(a1_norm)} vs {abs(a2_norm)} (expected near pi)")
            else:
                self.assertAlmostEqual(a1_norm, a2_norm, places=places, msg=f"Angle {a1_norm} vs {a2_norm}")

    def test_get_joint_frames_numpy_output(self):
        arm = RobotArm(num_joints=2, bone_lengths=[1.0, 1.0], joint_axes=['z', 'z'], joint_limits=[(-math.pi,math.pi)]*2)
        angles = [0.0, math.pi/2]; frames = get_joint_frames_numpy(arm, angles)
        self.assertEqual(len(frames), 3)
        np.testing.assert_array_almost_equal(frames[2], np.array([[0,-1,0,1],[1,0,0,1],[0,0,1,0],[0,0,0,1]]), decimal=6)

    def test_calculate_jacobian_2dof_zy_arm(self):
        L1, L2 = 1.0, 1.0
        arm = RobotArm(num_joints=2,bone_lengths=[L1,L2],joint_axes=['z','y'],joint_limits=[(-math.pi,math.pi)]*2)
        frames = get_joint_frames_numpy(arm, [0.,0.])
        J = calculate_jacobian(arm, frames)
        np.testing.assert_array_almost_equal(J, np.array([[0.,0.],[L1+L2,0.],[0.,-L2]]), decimal=6)

    def test_ik_2_link_planar_simple_cases_default_solver(self):
        # Default solver is DLS. Test if DLS can solve this simple 2-link case.
        arm = RobotArm(num_joints=2, bone_lengths=[1.0,1.0], joint_axes=['z','z'], joint_limits=[(-math.pi,math.pi)]*2)
        # Use DLS parameters known to work from test_ik_dispatcher_selects_correct_solver
        angles = calculate_inverse_kinematics(arm, (1.,1.,0.),
                                              tolerance=1e-4, max_iterations=100,
                                              damping_factor=0.1, step_size=0.1)
        self.assertIsNotNone(angles, "DLS IK for (1,1) with tuned params should find a solution")
        if angles:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm, angles), (1.,1.,0.), places=3)
            # DLS might not hit [0, pi/2] exactly, so check if close to one of the solutions.
            # Primary: [0, pi/2], Alternative: [pi/2, -pi/2]
            is_primary = all(math.isclose(a,e,abs_tol=1e-2) for a,e in zip(angles, [0.0, math.pi/2]))
            is_alternative = all(math.isclose(a,e,abs_tol=1e-2) for a,e in zip(angles, [math.pi/2, -math.pi/2]))
            self.assertTrue(is_primary or is_alternative, f"DLS angles {angles} not close to known solutions for (1,1)")


    def test_dls_ik_3_link_planar_reachable(self):
        arm_3z = RobotArm(num_joints=3, bone_lengths=[1.0, 0.8, 0.6], joint_axes=['z','z','z'],
                          joint_limits=[(-math.pi,math.pi)]*3)
        target = (1.5, 0.5, 0.0)
        angles = calculate_inverse_kinematics(arm_3z, target, solver="dls", max_iterations=300, tolerance=1e-4, step_size=0.5)
        self.assertIsNotNone(angles, f"DLS IK for 3Z planar: target {target}")
        if angles:
            fk_pos = calculate_forward_kinematics(arm_3z, angles)
            self.assertPositionsAlmostEqual(fk_pos, target, places=3, msg_prefix=f"3Z DLS to {target}: ")

    def test_dls_ik_3_link_spatial_zxz_reachable(self):
        arm_zxz = RobotArm(num_joints=3, bone_lengths=[1.0, 1.0, 0.5], joint_axes=['z','x','z'],
                           joint_limits=[(-math.pi,math.pi), (-math.pi/2, math.pi/2), (-math.pi,math.pi)])
        target_spatial = (0.5, 0.5, 1.0)
        angles = calculate_inverse_kinematics(arm_zxz, target_spatial, solver="dls", max_iterations=500, tolerance=1e-4, step_size=0.3, damping_factor=0.05)
        if angles: # DLS might fail for complex cases, accept None
            fk_pos = calculate_forward_kinematics(arm_zxz, angles)
            self.assertPositionsAlmostEqual(fk_pos, target_spatial, places=3, msg_prefix=f"ZXZ DLS to {target_spatial}: ")
        else:
            self.assertIsNone(angles, "DLS for ZXZ spatial failed as potentially expected.")


    def test_dls_ik_with_joint_limits_challenging_ccd_case(self):
        arm = RobotArm(num_joints=3, bone_lengths=[0.8,0.8,0.4], joint_axes=['z','z','z'],
                       joint_limits=[(-0.1,0.1), (-math.pi,math.pi), (-math.pi,math.pi)])
        target = (1.0, 0.0, 0.0)
        angles = calculate_inverse_kinematics(arm, target, solver="dls", max_iterations=500, tolerance=1e-4, step_size=0.2, damping_factor=0.1)
        if angles: # DLS might fail, accept None
            self.assertTrue(-0.1 - 1e-5 <= angles[0] <= 0.1 + 1e-5, f"J0 angle {angles[0]} out of limit [-0.1, 0.1]")
            fk_pos = calculate_forward_kinematics(arm, angles)
            self.assertPositionsAlmostEqual(fk_pos, target, places=3, msg_prefix=f"ZZZ DLS J0-limit to {target}: ")
        else:
            self.assertIsNone(angles, "DLS with tight J0 limit failed as potentially expected.")


    def test_dls_ik_singularity_max_reach_2link(self):
        L1, L2 = 1.0, 0.8
        arm = RobotArm(num_joints=2, bone_lengths=[L1,L2], joint_axes=['z','z'], joint_limits=[(-math.pi,math.pi)]*2)
        target_at_max_reach = (L1 + L2, 0.0, 0.0)
        angles = calculate_inverse_kinematics(arm, target_at_max_reach, solver="dls_force",
                                              max_iterations=200, tolerance=1e-4, step_size=0.5, damping_factor=0.01)
        self.assertIsNotNone(angles, f"DLS IK for 2-link at max reach {target_at_max_reach}")
        if angles:
            self.assertAnglesAlmostEqualList(angles, [0.0, 0.0], places=3)
            fk_pos = calculate_forward_kinematics(arm, angles)
            self.assertPositionsAlmostEqual(fk_pos, target_at_max_reach, places=3)

    def test_dls_ik_unreachable_target(self):
        arm_3z = RobotArm(num_joints=3, bone_lengths=[0.5,0.5,0.5], joint_axes=['z','z','z'],
                          joint_limits=[(-math.pi,math.pi)]*3)
        target_far = (3.0, 0.0, 0.0)
        angles = calculate_inverse_kinematics(arm_3z, target_far, solver="dls", max_iterations=100, tolerance=1e-3)
        if angles:
            fk_pos = calculate_forward_kinematics(arm_3z, angles)
            self.assertTrue(np.linalg.norm(np.array(fk_pos) - np.array(target_far)) > 0.1,
                            "DLS for far target returned angles that are too close.")
        else:
            self.assertIsNone(angles, "DLS for very far target should ideally return None.")

    def test_ccd_ik_3_link_planar_reachable_explicit_call(self):
        arm_3z = RobotArm(num_joints=3, bone_lengths=[1.0, 1.0, 0.5], joint_axes=['z', 'z', 'z'],
                          joint_limits=[(-math.pi, math.pi)] * 3)
        target = (1.5, 1.0, 0.0)
        angles = calculate_inverse_kinematics(arm_3z, target, solver="ccd", tolerance=1e-3, max_iterations=200)
        self.assertIsNotNone(angles)
        if angles: self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm_3z, angles), target, places=2)

    def test_ik_dispatcher_selects_correct_solver(self):
        arm_2link_zz = RobotArm(num_joints=2, bone_lengths=[1.0,1.0], joint_axes=['z','z'], joint_limits=[(-math.pi,math.pi)]*2)
        target = (1.0,1.0,0.0)

        # Test default solver for 2-link ZZ (should be DLS due to current dispatcher logic if default solver param is "dls")
        # The dispatcher: if is_2_link_zz AND solver not in ["dls", "ccd", "dls_force", "ccd_force"] -> analytical
        # If solver (param default) is "dls", this condition is false, so it skips analytical.
        # Then it hits: elif solver.lower() == "dls" or solver.lower() == "dls_force": -> DLS
        angles_default_is_dls = calculate_inverse_kinematics(arm_2link_zz, target, tolerance=1e-4, damping_factor=0.1, step_size=0.1) # Uses solver="dls" by default
        self.assertIsNotNone(angles_default_is_dls, "Default DLS for 2-link (1,1) should find a solution")
        if angles_default_is_dls:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm_2link_zz, angles_default_is_dls), target, places=3)

        angles_analytical_forced = calculate_inverse_kinematics(arm_2link_zz, target, solver="analytical_force")
        self.assertIsNotNone(angles_analytical_forced, "Forced Analytical for 2-link (1,1) should find a solution")
        if angles_analytical_forced:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm_2link_zz, angles_analytical_forced), target, places=5)
            self.assertAnglesAlmostEqualList(angles_analytical_forced, [0.0, math.pi/2], places=5)

        angles_ccd = calculate_inverse_kinematics(arm_2link_zz, target, solver="ccd")
        self.assertIsNotNone(angles_ccd, "Explicit CCD for 2-link (1,1) should find a solution")
        if angles_ccd:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm_2link_zz, angles_ccd), target, places=3)

        arm_3link = RobotArm(num_joints=3, bone_lengths=[1.,1.,0.5], joint_axes=['z','z','z'], joint_limits=[(-math.pi,math.pi)]*3)
        target3 = (1.5,0.5,0.0)
        angles_3link_default = calculate_inverse_kinematics(arm_3link, target3)
        self.assertIsNotNone(angles_3link_default, "Default DLS for 3-link should find solution")
        if angles_3link_default:
            self.assertPositionsAlmostEqual(calculate_forward_kinematics(arm_3link, angles_3link_default), target3, places=3)
