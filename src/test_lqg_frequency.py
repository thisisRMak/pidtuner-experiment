"""Tests for lqg_frequency.py."""

import unittest

import numpy as np

from plant import StateSpacePlant
from lqg_frequency import (loop_transfer_at_input, sensitivity_complementary,
                            compute_sensitivity)
from lqg_design_methods import LQGDesignResult, StateFeedbackGains


def _scalar_result(k=1.0):
    """A=[0], B=[1], C=[1], D=[0], K=[k] — integrator under proportional
    state feedback, full-state-feedback (no Kalman)."""
    plant = StateSpacePlant(A=np.array([[0.0]]), B=np.array([[1.0]]),
                            C=np.array([[1.0]]), D=np.array([[0.0]]))
    gains = StateFeedbackGains(K=np.array([[k]]))
    return LQGDesignResult(method="LQR", plant=plant, gains=gains,
                           S=np.array([[0.0]]), closed_loop_poles=np.array([-k]),
                           Q=np.array([[1.0]]), R=np.array([[1.0]]))


class TestLoopTransferAtInput(unittest.TestCase):
    def test_scalar_integrator_analytic(self):
        # L(s) = K/(s-A)*B = k/s. At s=1j*w, L = k/(1j*w).
        result = _scalar_result(k=1.0)
        L = loop_transfer_at_input(result, 1j * 1.0)
        expected = 1.0 / (1j * 1.0)
        self.assertAlmostEqual(L[0, 0], expected)


class TestSensitivityComplementary(unittest.TestCase):
    def test_S_plus_T_is_identity(self):
        L = np.array([[2.0 + 1.0j, 0.3], [0.1, 1.0 - 0.5j]])
        S, T = sensitivity_complementary(L)
        np.testing.assert_allclose(S + T, np.eye(2), atol=1e-12)

    def test_scalar_known_value(self):
        # k=1 integrator at w=1: L = 1/1j = -1j. S = 1/(1-1j) = (1+1j)/2.
        # |S| = 1/sqrt(2).
        L = np.array([[1.0 / (1j * 1.0)]])
        S, T = sensitivity_complementary(L)
        self.assertAlmostEqual(abs(S[0, 0]), 1.0 / np.sqrt(2), places=10)
        self.assertAlmostEqual(abs(T[0, 0]), 1.0 / np.sqrt(2), places=10)


class TestComputeSensitivity(unittest.TestCase):
    def test_peaks_match_manual_sweep(self):
        result = _scalar_result(k=1.0)
        omega = np.logspace(-2, 2, 50)
        res = compute_sensitivity(result, omega=omega)
        # Manually recompute peak sigma_S the slow way and compare.
        manual = []
        for w in omega:
            L = loop_transfer_at_input(result, 1j * w)
            S, _ = sensitivity_complementary(L)
            manual.append(abs(S[0, 0]))
        self.assertAlmostEqual(res.Ms, max(manual), places=10)
        self.assertEqual(res.loop_point, "plant_input")

    def test_default_omega_used_when_none(self):
        result = _scalar_result(k=1.0)
        res = compute_sensitivity(result)
        self.assertEqual(len(res.omega), 300)


if __name__ == "__main__":
    unittest.main()
