"""Unit tests for mimo_pi.py — MIMO PI control + multivariable integral
anti-windup (Windup_AEN 7.pdf §9.3).

Run with:
    python test_mimo_pi.py
or:
    python -m unittest test_mimo_pi -v
"""

import unittest

import numpy as np

from plant import StateSpacePlant
from mimo_pi import (
    MIMOPIGains, mimo_pi_step, simulate_mimo_pi,
    mimo_pi_closed_loop_poles, saturation_mask, _inv_or_pinv,
)


def coupled_plant():
    """A stable 2-state, 2-input, 2-output plant with input coupling (B has
    off-diagonal terms) — simple enough to hand-check, coupled enough that
    "treat each actuator independently" would be wrong."""
    A = [[-1.0, 0.0], [0.0, -2.0]]
    B = [[1.0, 0.5], [0.3, 1.0]]
    C = [[1.0, 0.0], [0.0, 1.0]]
    D = [[0.0, 0.0], [0.0, 0.0]]
    return StateSpacePlant(A=A, B=B, C=C, D=D)


def single_integrator():
    """A = 0, B = C = 1, D = 0 — the textbook single integrator, used to
    check mimo_pi_closed_loop_poles against the classic hand-derived
    PI-on-integrator result."""
    return StateSpacePlant(A=[[0.0]], B=[[1.0]], C=[[1.0]], D=[[0.0]])


# ─────────────────────────────────────────────────────────────────────────────
# MIMOPIGains
# ─────────────────────────────────────────────────────────────────────────────

class TestMIMOPIGains(unittest.TestCase):
    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            MIMOPIGains(KP=[[1, 0], [0, 1]], KI=[[1, 0, 0], [0, 1, 0]])

    def test_nu_ny_properties(self):
        gains = MIMOPIGains(KP=np.eye(2), KI=np.eye(2))
        self.assertEqual(gains.nu, 2)
        self.assertEqual(gains.ny, 2)


# ─────────────────────────────────────────────────────────────────────────────
# _inv_or_pinv
# ─────────────────────────────────────────────────────────────────────────────

class TestInvOrPinv(unittest.TestCase):
    def test_well_conditioned_uses_exact_inverse(self):
        M = np.array([[2.0, 0.0], [0.0, 2.0]])
        Minv, used_pinv = _inv_or_pinv(M)
        np.testing.assert_allclose(Minv, np.linalg.inv(M))
        self.assertFalse(used_pinv)

    def test_singular_falls_back_to_pinv(self):
        M = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank-deficient
        Minv, used_pinv = _inv_or_pinv(M)
        self.assertTrue(used_pinv)
        np.testing.assert_allclose(Minv, np.linalg.pinv(M))

    def test_nonsquare_falls_back_to_pinv(self):
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        Minv, used_pinv = _inv_or_pinv(M)
        self.assertTrue(used_pinv)


# ─────────────────────────────────────────────────────────────────────────────
# mimo_pi_step — direct, hand-checkable unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMimoPiStep(unittest.TestCase):
    def setUp(self):
        self.gains = MIMOPIGains(KP=np.eye(2), KI=np.eye(2))
        self.u_min = np.array([-1.0, -1.0])
        self.u_max = np.array([1.0, 1.0])
        self.zero = np.array([0.0, 0.0])

    def test_conditional_integrates_normally_when_not_saturated(self):
        u, integral = mimo_pi_step(self.gains, self.zero, r=[0.1, 0.05], y=[0.0, 0.0],
                                   dt=1.0, u_min=self.u_min, u_max=self.u_max,
                                   antiwindup="conditional")
        # e = [0.1, 0.05], trial_int = [0.1, 0.05], u_unsat = e + trial_int,
        # well within bounds -> no saturation -> plain integration.
        np.testing.assert_allclose(integral, [0.1, 0.05])
        np.testing.assert_allclose(u, [0.2, 0.1])

    def test_conditional_freezes_whole_vector_block_wise(self):
        # e = [10, 0.1] -> u_unsat = [20, 0.2] -> channel 0 saturates hard,
        # channel 1 (0.2) would NOT saturate on its own. A naive per-channel
        # freeze would still integrate channel 1; this module's block-wise
        # policy (freeze the whole xI vector whenever ANY channel
        # saturates) must leave channel 1's integral untouched too — that's
        # the entire point of doing this at the MIMO level (see module
        # docstring / Windup_AEN 7.pdf's opening warning about coupling).
        u, integral = mimo_pi_step(self.gains, self.zero, r=[10.0, 0.1], y=[0.0, 0.0],
                                   dt=1.0, u_min=self.u_min, u_max=self.u_max,
                                   antiwindup="conditional")
        np.testing.assert_allclose(u, [1.0, 0.2])
        np.testing.assert_allclose(integral, [0.0, 0.0])

    def test_resettable_lands_exactly_on_usat(self):
        # eq. 9.121-9.123: after the reset, recomputing Uc from the SAME e
        # and the new xI must reproduce u_sat exactly.
        KI_inv, _ = _inv_or_pinv(self.gains.KI)
        e = np.array([10.0, 0.1])
        u, integral = mimo_pi_step(self.gains, self.zero, r=e.tolist(), y=[0.0, 0.0],
                                   dt=1.0, u_min=self.u_min, u_max=self.u_max,
                                   antiwindup="resettable", inv=KI_inv)
        np.testing.assert_allclose(u, [1.0, 0.2])
        recomputed = self.gains.KP @ e + self.gains.KI @ integral
        np.testing.assert_allclose(recomputed, u, atol=1e-10)

    def test_resettable_integrates_normally_when_not_saturated(self):
        KI_inv, _ = _inv_or_pinv(self.gains.KI)
        u, integral = mimo_pi_step(self.gains, self.zero, r=[0.1, 0.05], y=[0.0, 0.0],
                                   dt=1.0, u_min=self.u_min, u_max=self.u_max,
                                   antiwindup="resettable", inv=KI_inv)
        np.testing.assert_allclose(integral, [0.1, 0.05])
        np.testing.assert_allclose(u, [0.2, 0.1])

    def test_hanus_vanishes_when_not_saturated(self):
        KP_inv, _ = _inv_or_pinv(self.gains.KP)
        u, integral = mimo_pi_step(self.gains, self.zero, r=[0.1, 0.05], y=[0.0, 0.0],
                                   dt=1.0, u_min=self.u_min, u_max=self.u_max,
                                   antiwindup="hanus", inv=KP_inv)
        # No saturation -> correction term is zero -> same as plain integration.
        np.testing.assert_allclose(integral, [0.1, 0.05])
        np.testing.assert_allclose(u, [0.2, 0.1])

    def test_hanus_corrects_when_saturated(self):
        KP_inv, _ = _inv_or_pinv(self.gains.KP)
        e = np.array([10.0, 0.1])
        u, integral = mimo_pi_step(self.gains, self.zero, r=e.tolist(), y=[0.0, 0.0],
                                   dt=1.0, u_min=self.u_min, u_max=self.u_max,
                                   antiwindup="hanus", inv=KP_inv)
        np.testing.assert_allclose(u, [1.0, 0.2])
        # Hand-derived: trial_int = e, u_unsat = 2e, correction = KP_inv@(u-u_unsat)
        expected = e + KP_inv @ (u - 2 * e)
        np.testing.assert_allclose(integral, expected, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# mimo_pi_closed_loop_poles
# ─────────────────────────────────────────────────────────────────────────────

class TestClosedLoopPoles(unittest.TestCase):
    def test_matches_classic_pi_on_integrator(self):
        # G(s) = 1/s under Uc = KP*E + KI*int(E), KP=2, KI=1: closed-loop
        # char. eq. s^2 + KP s + KI = s^2 + 2s + 1 = (s+1)^2 -> poles at -1,-1.
        plant = single_integrator()
        gains = MIMOPIGains(KP=[[2.0]], KI=[[1.0]])
        poles = mimo_pi_closed_loop_poles(plant, gains)
        np.testing.assert_allclose(sorted(np.real(poles)), [-1.0, -1.0], atol=1e-9)

    def test_coupled_plant_is_stabilizable(self):
        plant = coupled_plant()
        gains = MIMOPIGains(KP=2.0 * np.eye(2), KI=1.0 * np.eye(2))
        poles = mimo_pi_closed_loop_poles(plant, gains)
        self.assertTrue(np.all(np.real(poles) < -1e-9))

    def test_rejects_mismatched_gains_directly(self):
        # Validation must not be limited to simulate_mimo_pi — a caller
        # going straight to mimo_pi_closed_loop_poles gets the same check.
        plant = coupled_plant()
        gains = MIMOPIGains(KP=np.eye(3), KI=np.eye(3))
        with self.assertRaises(ValueError):
            mimo_pi_closed_loop_poles(plant, gains)


# ─────────────────────────────────────────────────────────────────────────────
# simulate_mimo_pi — end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateMimoPi(unittest.TestCase):
    def setUp(self):
        self.plant = coupled_plant()
        self.gains = MIMOPIGains(KP=2.0 * np.eye(2), KI=1.0 * np.eye(2))

    def test_non_square_plant_rejected(self):
        plant = StateSpacePlant(A=[[-1.0]], B=[[1.0, 0.5]],
                                C=[[1.0]], D=[[0.0, 0.0]])
        gains = MIMOPIGains(KP=[[1.0, 0.0]], KI=[[1.0, 0.0]])
        with self.assertRaises(ValueError):
            simulate_mimo_pi(plant, gains)

    def test_gains_shape_must_match_plant(self):
        gains = MIMOPIGains(KP=np.eye(3), KI=np.eye(3))
        with self.assertRaises(ValueError):
            simulate_mimo_pi(self.plant, gains)

    def test_default_is_conditional_and_tracks_step(self):
        sim = simulate_mimo_pi(self.plant, self.gains, u_min=[-1e6, -1e6],
                               u_max=[1e6, 1e6])
        self.assertEqual(sim.antiwindup, "conditional")
        self.assertTrue(sim.stable)
        # Unit step on both channels, wide-open actuators -> should settle
        # near r = [1, 1].
        np.testing.assert_allclose(sim.y[-1], [1.0, 1.0], atol=1e-2)

    def test_modes_identical_when_never_saturated(self):
        common = dict(plant=self.plant, gains=self.gains,
                      u_min=[-1e6, -1e6], u_max=[1e6, 1e6])
        cond = simulate_mimo_pi(antiwindup="conditional", **common)
        hanus = simulate_mimo_pi(antiwindup="hanus", **common)
        reset = simulate_mimo_pi(antiwindup="resettable", **common)
        np.testing.assert_allclose(cond.y, hanus.y, atol=1e-8)
        np.testing.assert_allclose(cond.y, reset.y, atol=1e-8)
        np.testing.assert_allclose(cond.u, hanus.u, atol=1e-8)
        np.testing.assert_allclose(cond.u, reset.u, atol=1e-8)

    def test_modes_diverge_when_saturated(self):
        # Tight enough to saturate hard early, loose enough that the loop
        # actually desaturates and recovers within the sim horizon — that
        # recovery phase is exactly where conditional-freeze (still
        # unwinding residual integral) and resettable-reset (already
        # landed exactly on u_sat, eq. 9.123) must diverge. Bounds tight
        # enough to pin the actuator for the whole horizon make both modes
        # trivially identical (u is just the clipped constant either way).
        common = dict(plant=self.plant, gains=self.gains,
                      u_min=[-0.6, -0.6], u_max=[0.6, 0.6])
        cond = simulate_mimo_pi(antiwindup="conditional", **common)
        reset = simulate_mimo_pi(antiwindup="resettable", **common)
        self.assertTrue(saturation_mask(cond).any())
        self.assertGreater(np.max(np.abs(cond.y - reset.y)), 1e-6)

    def test_pinv_flag_surfaces_on_singular_gains(self):
        singular_KI = MIMOPIGains(KP=2.0 * np.eye(2), KI=np.zeros((2, 2)))
        sim = simulate_mimo_pi(self.plant, singular_KI,
                               u_min=[-0.05, -0.05], u_max=[0.05, 0.05],
                               antiwindup="resettable")
        self.assertTrue(sim.pinv_used)

    def test_pinv_flag_false_when_gains_well_conditioned(self):
        sim = simulate_mimo_pi(self.plant, self.gains,
                               u_min=[-0.05, -0.05], u_max=[0.05, 0.05],
                               antiwindup="resettable")
        self.assertFalse(sim.pinv_used)

    def test_custom_r_trace_shape(self):
        t = np.arange(0.0, 2.0, 0.01)
        r = np.tile(np.array([1.0, -0.5]), (len(t), 1))
        sim = simulate_mimo_pi(self.plant, self.gains, t=t, r=r,
                               u_min=[-1e6, -1e6], u_max=[1e6, 1e6])
        np.testing.assert_allclose(sim.r, r)

    def test_saturation_mask_shape(self):
        sim = simulate_mimo_pi(self.plant, self.gains,
                               u_min=[-0.05, -0.05], u_max=[0.05, 0.05])
        mask = saturation_mask(sim)
        self.assertEqual(mask.shape, sim.u.shape)
        self.assertTrue(mask.any())

    def test_metrics_report_unstable_on_divergence(self):
        # KI with a positive-feedback sign flips stability for this plant.
        bad_gains = MIMOPIGains(KP=2.0 * np.eye(2), KI=-5.0 * np.eye(2))
        sim = simulate_mimo_pi(self.plant, bad_gains,
                               t=np.arange(0.0, 5.0, 0.01))
        self.assertFalse(sim.stable)


if __name__ == "__main__":
    unittest.main()
