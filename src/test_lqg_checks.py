"""Unit tests for lqg_checks.py — see docs/lqg_testing.md for what each
check means. Two kinds of coverage here: (1) each check function in
isolation, fed deliberately-good and deliberately-bad inputs so both the
PASS and FAIL branches are exercised; (2) checks_for_result() against real
design results from lqg_design_methods.py, which doubles as a cross-check
that its Aaug/Baug/Q̂ reconstruction for ExplicitModelFollowingResult
matches what ExplicitModelFollowing.design() actually solved with.
"""

from __future__ import annotations

import unittest

import numpy as np

from plant import StateSpacePlant
from lqg_design_methods import LQR, LQG
from lqg_implicit import ImplicitModelFollowing
from lqg_explicit import ExplicitModelFollowing
from lqg_checks import (
    precheck_Q_R, precheck_stabilizability, precheck_detectability,
    postcheck_are_residual, postcheck_symmetric_psd, postcheck_poles_stable,
    checks_for_result,
)


def double_integrator():
    return StateSpacePlant.from_matrices(A=[[0, 1], [0, 0]], B=[[0], [1]], C=[[1, 0]])


class TestPrecheckQR(unittest.TestCase):
    def test_well_posed_QR_all_pass(self):
        results = precheck_Q_R(np.eye(2), np.eye(1))
        self.assertTrue(all(r.passed for r in results))

    def test_asymmetric_Q_fails(self):
        Q = np.array([[1.0, 0.5], [0.0, 1.0]])
        results = precheck_Q_R(Q, np.eye(1))
        by_name = {r.name: r.passed for r in results}
        self.assertFalse(by_name["Q symmetric"])

    def test_indefinite_Q_fails_psd(self):
        Q = np.diag([1.0, -1.0])
        results = precheck_Q_R(Q, np.eye(2))
        by_name = {r.name: r.passed for r in results}
        self.assertFalse(by_name["Q positive semi-definite"])

    def test_singular_R_fails_pd(self):
        R = np.diag([1.0, 0.0])
        results = precheck_Q_R(np.eye(2), R)
        by_name = {r.name: r.passed for r in results}
        self.assertFalse(by_name["R positive definite"])


class TestPrecheckStabilizability(unittest.TestCase):
    def test_controllable_pair_passes(self):
        A = np.diag([-1.0, -2.0])
        B = np.array([[1.0], [1.0]])
        self.assertTrue(precheck_stabilizability(A, B).passed)

    def test_uncontrollable_but_stable_mode_still_passes(self):
        # mode 2 (-2) is uncontrollable, but it's stable, so LQR doesn't
        # need to move it — stabilizability should still hold.
        A = np.diag([-1.0, -2.0])
        B = np.array([[1.0], [0.0]])
        self.assertTrue(precheck_stabilizability(A, B).passed)

    def test_uncontrollable_unstable_mode_fails(self):
        # mode 1 (+1, unstable) is uncontrollable — not stabilizable.
        A = np.diag([1.0, -2.0])
        B = np.array([[0.0], [1.0]])
        self.assertFalse(precheck_stabilizability(A, B).passed)


class TestPrecheckDetectability(unittest.TestCase):
    def test_observable_pair_passes(self):
        A = np.diag([-1.0, -2.0])
        Q = np.eye(2)
        self.assertTrue(precheck_detectability(A, Q).passed)

    def test_unobservable_but_stable_mode_still_passes(self):
        A = np.diag([-1.0, -2.0])
        Q = np.diag([1.0, 0.0])  # mode 2 unpenalized, but stable
        self.assertTrue(precheck_detectability(A, Q).passed)

    def test_unobservable_unstable_mode_fails(self):
        A = np.diag([1.0, -2.0])
        Q = np.diag([0.0, 1.0])  # mode 1 (+1, unstable) unpenalized
        self.assertFalse(precheck_detectability(A, Q).passed)


class TestPostcheckAREResidual(unittest.TestCase):
    def test_correct_S_passes(self):
        p = double_integrator()
        res = LQR(p, Q=np.eye(2), R=np.eye(1)).design()
        check = postcheck_are_residual(p.A, p.B, res.Q, res.R, res.S)
        self.assertTrue(check.passed)

    def test_perturbed_S_fails(self):
        p = double_integrator()
        res = LQR(p, Q=np.eye(2), R=np.eye(1)).design()
        S_wrong = res.S + np.eye(2)  # not a Riccati solution anymore
        check = postcheck_are_residual(p.A, p.B, res.Q, res.R, S_wrong)
        self.assertFalse(check.passed)


class TestPostcheckSymmetricPSD(unittest.TestCase):
    def test_identity_passes(self):
        results = postcheck_symmetric_psd(np.eye(3), "M")
        self.assertTrue(all(r.passed for r in results))

    def test_asymmetric_fails(self):
        M = np.array([[1.0, 2.0], [0.0, 1.0]])
        results = postcheck_symmetric_psd(M, "M")
        by_name = {r.name: r.passed for r in results}
        self.assertFalse(by_name["M symmetric"])

    def test_negative_definite_fails_psd(self):
        M = -np.eye(2)
        results = postcheck_symmetric_psd(M, "M")
        by_name = {r.name: r.passed for r in results}
        self.assertFalse(by_name["M positive semi-definite"])


class TestPostcheckPolesStable(unittest.TestCase):
    def test_stable_poles_pass(self):
        self.assertTrue(postcheck_poles_stable([-1, -2 + 3j, -2 - 3j], "poles").passed)

    def test_unstable_pole_fails(self):
        self.assertFalse(postcheck_poles_stable([-1, 0.5], "poles").passed)

    def test_nonfinite_pole_fails(self):
        self.assertFalse(postcheck_poles_stable([-1, float("nan")], "poles").passed)


class TestChecksForResult(unittest.TestCase):
    def test_lqr_result_all_pass(self):
        p = double_integrator()
        res = LQR(p, Q=np.eye(2), R=np.eye(1)).design()
        out = checks_for_result(res)
        self.assertTrue(all(c.passed for c in out["pre"]))
        self.assertTrue(all(c.passed for c in out["post"]))
        self.assertNotIn("kalman_pre", out)

    def test_lqg_result_includes_kalman_checks(self):
        p = double_integrator()
        res = LQG(p, Q=np.eye(2), R=np.eye(1), Qw=0.01 * np.eye(2), Rv=0.1 * np.eye(1)).design()
        out = checks_for_result(res)
        self.assertTrue(all(c.passed for c in out["pre"]))
        self.assertTrue(all(c.passed for c in out["post"]))
        self.assertTrue(all(c.passed for c in out["kalman_pre"]))
        self.assertTrue(all(c.passed for c in out["kalman_post"]))

    def test_implicit_model_following_result(self):
        from test_lqg import electro_mechanical_system
        p = electro_mechanical_system()
        Am = np.array([[-0.1, 0], [0, -0.07]])
        res = ImplicitModelFollowing(p, Am=Am, Q1=np.eye(2), R=np.eye(2)).design()
        out = checks_for_result(res)
        self.assertTrue(all(c.passed for c in out["pre"]))
        self.assertTrue(all(c.passed for c in out["post"]))

    def test_explicit_model_following_result(self):
        from test_lqg import electro_mechanical_system
        p = electro_mechanical_system()
        Am = np.array([[-0.1, 0], [0, -0.07]])
        res = ExplicitModelFollowing(p, Am=Am, Q1=np.eye(2), R=np.eye(2)).design()
        out = checks_for_result(res)
        # This ARE residual check re-derives Aaug/Baug/Qhat independently
        # of ExplicitModelFollowing.design() — if it passes, that's a
        # cross-check that both code paths agree, not just that the design
        # itself is self-consistent.
        self.assertTrue(all(c.passed for c in out["post"]))


if __name__ == "__main__":
    unittest.main()
