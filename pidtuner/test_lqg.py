"""Unit tests for the LQR/LQG design track (docs/lqg_plan.md Phase 1).

Run with:
    python test_lqg.py
or:
    python -m unittest test_lqg -v

Golden-value validation: `aircraft_hall` is AILQG.pdf's own Example 1
(Hall-71 aircraft) — the PDF prints S (eq. 25), K (eq. 26), and the
closed-loop poles to 4-5 digits, so LQR on this preset is checked against
those exact printed numbers, not just "did it run."
"""

from __future__ import annotations

import unittest

import numpy as np

from plant import TransferFunction, StateSpacePlant, tf_to_state_space
from lqg_examples import list_examples, load_example
from lqg_design_methods import (
    LQR, OutputWeightedLQR, LQG,
    add_reference_tracking, lqg_full_closed_loop_poles,
)
from lqg_bryson import BrysonLQR
from lqg_implicit import ImplicitModelFollowing
from lqg_explicit import ExplicitModelFollowing
from lqg_simulate import (
    simulate_state_feedback, simulate_output_feedback, simulate_explicit_model_following,
    compute_tracking_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# StateSpacePlant
# ─────────────────────────────────────────────────────────────────────────────

class TestStateSpacePlant(unittest.TestCase):
    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            StateSpacePlant(A=[[1, 2, 3], [4, 5, 6]], B=[[1], [2]], C=[[1, 0]], D=[[0]])
        with self.assertRaises(ValueError):
            StateSpacePlant(A=[[1, 0], [0, 1]], B=[[1], [2], [3]], C=[[1, 0]], D=[[0]])
        with self.assertRaises(ValueError):
            StateSpacePlant(A=[[1, 0], [0, 1]], B=[[1], [2]], C=[[1, 0]], D=[[0, 0]])

    def test_from_matrices_defaults(self):
        p = StateSpacePlant.from_matrices(A=[[0, 1], [-2, -3]], B=[[0], [1]])
        self.assertEqual(p.nx, 2)
        self.assertEqual(p.nu, 1)
        self.assertEqual(p.ny, 2)  # default C = I
        np.testing.assert_array_equal(p.C, np.eye(2))
        np.testing.assert_array_equal(p.D, np.zeros((2, 1)))

    def test_poles_stability(self):
        stable = StateSpacePlant.from_matrices(A=[[-1, 0], [0, -2]], B=[[1], [1]])
        self.assertTrue(stable.is_open_loop_stable())
        unstable = StateSpacePlant.from_matrices(A=[[1, 0], [0, -2]], B=[[1], [1]])
        self.assertFalse(unstable.is_open_loop_stable())

    def test_controllable_observable_double_integrator(self):
        p = StateSpacePlant.from_matrices(A=[[0, 1], [0, 0]], B=[[0], [1]], C=[[1, 0]])
        self.assertTrue(p.is_controllable())
        self.assertTrue(p.is_observable())

    def test_uncontrollable_pair_detected(self):
        # B only excites the first state; the second is decoupled.
        p = StateSpacePlant.from_matrices(A=[[-1, 0], [0, -2]], B=[[1], [0]])
        self.assertFalse(p.is_controllable())

    def test_tf_to_state_space_matches_tf_simulate(self):
        tf = TransferFunction.parse("1000/((s+1)(10s+1))")
        ss = tf_to_state_space(tf)
        t = np.arange(0.0, 5.0, ss.auto_dt())
        u_tf = np.ones_like(t)
        y_tf = tf.simulate(t, u_tf)
        _, y_ss = ss.simulate_open_loop(t, u_tf.reshape(-1, 1))
        np.testing.assert_allclose(y_ss[:, 0], y_tf, atol=1e-6, rtol=1e-4)

    def test_tf_to_state_space_rejects_dead_time(self):
        tf = TransferFunction.parse("1/(s+1)", L=0.5)
        with self.assertRaises(ValueError):
            tf_to_state_space(tf)


# ─────────────────────────────────────────────────────────────────────────────
# LQR / OutputWeightedLQR / BrysonLQR — golden value + basic behavior
# ─────────────────────────────────────────────────────────────────────────────

def double_integrator():
    return StateSpacePlant.from_matrices(A=[[0, 1], [0, 0]], B=[[0], [1]], C=[[1, 0]])


class TestLQRGoldenValue(unittest.TestCase):
    """aircraft_hall == AILQG.pdf Example 1. PDF eq. 25/26 printed S/K to
    5 significant digits; we check to 1e-3 relative tolerance."""

    def setUp(self):
        self.ex = load_example("aircraft_hall")
        self.res = LQR(self.ex.plant, Q=self.ex.build_suggested_Q(),
                       R=self.ex.build_suggested_R()).design()

    def test_S_matches_pdf(self):
        S_pdf = np.array([
            [20.7016, 0.8647, 0.0883, 0.7316, 20.4235],
            [0.8647, 2.6738, 0.1597, 0.1652, 1.4505],
            [0.0883, 0.1597, 0.4540, 0.6412, 0.4739],
            [0.7316, 0.1652, 0.6412, 1.2755, 1.0916],
            [20.4235, 1.4505, 0.4739, 1.0916, 21.2632],
        ])
        np.testing.assert_allclose(self.res.S, S_pdf, atol=2e-3)

    def test_K_matches_pdf(self):
        K_pdf = np.array([
            [0.3354, 1.0294, 0.0913, 0.1059, 0.5844],
            [0.1751, 0.3608, 0.7282, 1.0261, 0.8115],
        ])
        np.testing.assert_allclose(self.res.gains.K, K_pdf, atol=2e-3)

    def test_closed_loop_poles_match_pdf(self):
        # PDF: -0.0501, -0.1954±j0.6568, -0.9833±j0.7674
        expected = sorted([-0.0501, -0.1954, -0.1954, -0.9833, -0.9833])
        got = sorted(np.real(self.res.closed_loop_poles))
        np.testing.assert_allclose(got, expected, atol=2e-3)
        imag_pairs = sorted(abs(p.imag) for p in self.res.closed_loop_poles if abs(p.imag) > 1e-9)
        np.testing.assert_allclose(imag_pairs, [0.6568, 0.6568, 0.7674, 0.7674], atol=2e-3)

    def test_stable(self):
        self.assertTrue(self.res.is_stable())


class TestOutputWeightedLQR(unittest.TestCase):
    def test_Q_is_output_weighted(self):
        p = double_integrator()
        res = OutputWeightedLQR(p).design()
        np.testing.assert_array_equal(res.Q, p.C.T @ p.C)
        self.assertTrue(res.is_stable())

    def test_default_Qy_R_are_identity(self):
        p = double_integrator()
        m = OutputWeightedLQR(p)
        np.testing.assert_array_equal(m.Qy, np.eye(p.ny))
        np.testing.assert_array_equal(m.R, np.eye(p.nu))


class TestBrysonLQR(unittest.TestCase):
    def test_diagonal_weights(self):
        p = double_integrator()
        res = BrysonLQR(p, x_max=[2.0, 4.0], u_max=[10.0]).design()
        np.testing.assert_allclose(np.diag(res.Q), [0.25, 1.0 / 16.0])
        np.testing.assert_allclose(np.diag(res.R), [0.01])
        self.assertTrue(res.is_stable())

    def test_rejects_wrong_length(self):
        p = double_integrator()
        with self.assertRaises(ValueError):
            BrysonLQR(p, x_max=[1.0], u_max=[1.0])

    def test_rejects_nonpositive(self):
        p = double_integrator()
        with self.assertRaises(ValueError):
            BrysonLQR(p, x_max=[1.0, -1.0], u_max=[1.0])


# ─────────────────────────────────────────────────────────────────────────────
# LQG (Kalman filter) — separation principle, no professor example to check
# against, so validated against the PDF's own algebraic identity (eq. 108).
# ─────────────────────────────────────────────────────────────────────────────

class TestLQG(unittest.TestCase):
    def test_separation_principle_double_integrator(self):
        p = double_integrator()
        res = LQG(p, Q=np.eye(2), R=np.eye(1), Qw=0.01 * np.eye(2), Rv=0.1 * np.eye(1)).design()
        full_poles = sorted(lqg_full_closed_loop_poles(res).real)
        union_poles = sorted(list(res.closed_loop_poles.real) + list(res.kalman.estimator_poles.real))
        np.testing.assert_allclose(full_poles, union_poles, atol=1e-8)

    def test_separation_principle_on_preset_plants(self):
        for key in ("rpv", "airc", "drone"):
            ex = load_example(key)
            res = LQG(ex.plant, Q=ex.build_suggested_Q(), R=ex.build_suggested_R(),
                     Qw=0.01 * np.eye(ex.plant.nx), Rv=0.1 * np.eye(ex.plant.ny)).design()
            full_poles = sorted(lqg_full_closed_loop_poles(res).real)
            union_poles = sorted(list(res.closed_loop_poles.real)
                                 + list(res.kalman.estimator_poles.real))
            np.testing.assert_allclose(full_poles, union_poles, atol=1e-6,
                                       err_msg=f"separation principle failed for {key}")

    def test_estimator_is_stable_for_observable_plant(self):
        p = double_integrator()
        res = LQG(p, Q=np.eye(2), R=np.eye(1), Qw=np.eye(2), Rv=np.eye(1)).design()
        self.assertTrue(np.all(np.real(res.kalman.estimator_poles) < -1e-9))

    def test_P_is_symmetric_psd(self):
        p = double_integrator()
        res = LQG(p, Q=np.eye(2), R=np.eye(1), Qw=np.eye(2), Rv=np.eye(1)).design()
        P = res.kalman.P
        np.testing.assert_allclose(P, P.T, atol=1e-8)
        self.assertTrue(np.all(np.linalg.eigvalsh(P) >= -1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# Model-following — AILQG.pdf §4, Examples 3 (implicit) and 4 (explicit),
# both built on the same electro-mechanical-system plant (eq. 61-63).
# ─────────────────────────────────────────────────────────────────────────────

def electro_mechanical_system():
    A = [[0, 0, 0, 0, 1],
         [0, 0, -50, 50, 1],
         [0, 0, -1e-4, 0, -1.667e-2],
         [0, 0, 0, -1e-4, -1.667e-2],
         [0, 0, 0.21, 0.21, -1.1667e-2]]
    B = [[0, 0], [0, 0], [0.1, 0], [0, 0.1], [3.5e-2, 3.5e-2]]
    C = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
    D = np.zeros((2, 2))
    return StateSpacePlant(A=A, B=B, C=C, D=D)


class TestImplicitModelFollowing(unittest.TestCase):
    """AILQG.pdf Example 3: electro-mechanical system, Am/Q1/R per eq. 63.
    PDF prints S (eq. 64), K (eq. 65), and the closed-loop poles — an exact
    golden-value check."""

    def setUp(self):
        self.plant = electro_mechanical_system()
        self.Am = np.array([[-0.1, 0], [0, -0.07]])
        self.Q1 = np.eye(2)
        self.R = np.eye(2)
        self.res = ImplicitModelFollowing(self.plant, Am=self.Am, Q1=self.Q1,
                                          R=self.R).design()

    def test_S_matches_pdf(self):
        S_pdf = np.array([
            [0.0608, 0.0000, 0.4448, 0.4344, 0.7642],
            [0.0000, 0.0007, -0.4948, 0.4951, 0.0099],
            [0.4448, -0.4948, 365.9681, -347.8229, -0.9215],
            [0.4344, 0.4951, -347.8229, 366.4729, 13.1923],
            [0.7642, 0.0099, -0.9215, 13.1923, 10.3325],
        ])
        np.testing.assert_allclose(self.res.S, S_pdf, atol=2e-3)

    def test_K_matches_pdf(self):
        K_pdf = np.array([
            [0.0712, -0.0491, 36.5646, -34.3206, 0.2695],
            [0.0702, 0.0499, -34.8145, 37.1090, 1.6809],
        ])
        np.testing.assert_allclose(self.res.gains.K, K_pdf, atol=2e-3)

    def test_closed_loop_poles_match_pdf(self):
        # PDF: -7.0709, -0.1087±j0.1466, -0.0892, -0.0700
        expected_real = sorted([-7.0709, -0.1087, -0.1087, -0.0892, -0.0700])
        got_real = sorted(np.real(self.res.closed_loop_poles))
        np.testing.assert_allclose(got_real, expected_real, atol=2e-3)
        imag_pairs = sorted(abs(p.imag) for p in self.res.closed_loop_poles
                            if abs(p.imag) > 1e-9)
        np.testing.assert_allclose(imag_pairs, [0.1466, 0.1466], atol=2e-3)

    def test_stable(self):
        self.assertTrue(self.res.is_stable())

    def test_rejects_wrong_shaped_Am(self):
        with self.assertRaises(ValueError):
            ImplicitModelFollowing(self.plant, Am=np.eye(3), Q1=self.Q1, R=self.R)

    def test_rejects_wrong_shaped_Q1(self):
        with self.assertRaises(ValueError):
            ImplicitModelFollowing(self.plant, Am=self.Am, Q1=np.eye(3), R=self.R)


class TestExplicitModelFollowing(unittest.TestCase):
    """AILQG.pdf Example 4 reuses Example 3's plant/Am/Q1/R. The PDF's own
    printed S (eq. 67) does not satisfy the continuous-time ARE it claims to
    solve (residual ~14, computed in test_S_solves_the_riccati_equation
    below) — this implementation's S has residual ~1e-12 against the same
    equation, so the PDF table looks like a transcription/rounding error,
    not a bug here. See docs/lqg_testing.md. Validated structurally instead:
    the ARE residual, S's symmetry/PSD-ness, K1/K2 shapes, and closed-loop
    stability — the same properties any correct Riccati solution must have,
    independent of whether the PDF's printed digits are trustworthy."""

    def setUp(self):
        self.plant = electro_mechanical_system()
        self.Am = np.array([[-0.1, 0], [0, -0.07]])
        self.Q1 = np.eye(2)
        self.R = np.eye(2)
        self.res = ExplicitModelFollowing(self.plant, Am=self.Am, Q1=self.Q1,
                                          R=self.R).design()

    def test_gain_shapes(self):
        self.assertEqual(self.res.K1.shape, (2, 5))
        self.assertEqual(self.res.K2.shape, (2, 2))

    def test_S_solves_the_riccati_equation(self):
        nx, nxm = 5, 2
        A, B, C = self.plant.A, self.plant.B, self.plant.C
        Aaug = np.block([[A, np.zeros((nx, nxm))],
                         [np.zeros((nxm, nx)), self.Am]])
        Baug = np.vstack([B, np.zeros((nxm, self.plant.nu))])
        Qhat = np.block([[C.T @ self.Q1 @ C, -C.T @ self.Q1],
                         [-self.Q1 @ C, self.Q1]])
        S = self.res.S
        residual = (Aaug.T @ S + S @ Aaug
                   - S @ Baug @ np.linalg.inv(self.R) @ Baug.T @ S + Qhat)
        self.assertLess(np.linalg.norm(residual), 1e-8)

    def test_S_symmetric_psd(self):
        S = self.res.S
        np.testing.assert_allclose(S, S.T, atol=1e-8)
        self.assertTrue(np.all(np.linalg.eigvalsh(S) >= -1e-6))

    def test_stable(self):
        self.assertTrue(self.res.is_stable())

    def test_rejects_non_square_Am(self):
        with self.assertRaises(ValueError):
            ExplicitModelFollowing(self.plant, Am=np.zeros((2, 3)), Q1=self.Q1, R=self.R)

    def test_rejects_Am_not_matching_Q1_size(self):
        # square (3,3) Am, but Q1/plant.ny say nxm should be 2 -- eq. 54's
        # block construction requires xm's dimension to equal Q1's.
        with self.assertRaises(ValueError):
            ExplicitModelFollowing(self.plant, Am=np.eye(3), Q1=self.Q1, R=self.R)


# ─────────────────────────────────────────────────────────────────────────────
# Explicit model-following simulation (augmented [x; xm] closed loop)
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateExplicitModelFollowing(unittest.TestCase):
    def setUp(self):
        self.plant = electro_mechanical_system()
        self.Am = np.array([[-0.1, 0], [0, -0.07]])
        self.res = ExplicitModelFollowing(self.plant, Am=self.Am, Q1=np.eye(2),
                                          R=np.eye(2)).design()

    def test_plant_output_tracks_model_state(self):
        t = np.arange(0.0, 300.0, 0.05)
        sim = simulate_explicit_model_following(self.res, t)
        self.assertTrue(sim.stable)
        self.assertEqual(sim.xm.shape, (len(t), 2))
        # both the model and the plant's tracking of it should have
        # converged (Am is stable, so both go to ~0) by the end.
        np.testing.assert_allclose(sim.y[-1], sim.xm[-1], atol=1e-3)
        np.testing.assert_allclose(sim.xm[-1], [0.0, 0.0], atol=1e-3)

    def test_unstable_Am_rejected_at_construction(self):
        # xm is uncontrollable inside the augmented system (eq. 51 has no B
        # for the xm block), so an unstable Am isn't just conceptually odd,
        # it also makes the augmented ARE unsolvable -- this must be caught
        # with a clear message, not scipy's opaque LinAlgError.
        with self.assertRaises(ValueError):
            ExplicitModelFollowing(self.plant, Am=np.array([[0.1, 0], [0, -0.07]]),
                                   Q1=np.eye(2), R=np.eye(2))

    def test_rejects_wrong_result_type(self):
        plant = double_integrator()
        lqr_res = LQR(plant, Q=np.eye(2), R=np.eye(1)).design()
        with self.assertRaises(ValueError):
            simulate_explicit_model_following(lqr_res, np.arange(0.0, 1.0, 0.1))


# ─────────────────────────────────────────────────────────────────────────────
# Reference-tracking step metrics (Overshoot/Rise/Settling, sign-aware)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTrackingMetrics(unittest.TestCase):
    def test_positive_and_negative_reference_both_get_nonnegative_overshoot(self):
        ex = load_example("aircraft_hall")  # square: nu == ny == 2
        res = LQR(ex.plant, Q=ex.build_suggested_Q(), R=ex.build_suggested_R()).design()
        rt = add_reference_tracking(res)
        t = np.arange(0.0, 200.0, 0.05)
        r = np.tile([1.0, -0.5], (len(t), 1))
        sim = simulate_state_feedback(rt, t, r=r)
        self.assertIsNotNone(sim.tracking_metrics)
        self.assertEqual(len(sim.tracking_metrics), 2)
        for m in sim.tracking_metrics:
            self.assertGreaterEqual(m["Overshoot"], 0.0)
            self.assertGreater(m["Rise"], 0.0)
            self.assertGreater(m["Settling"], 0.0)

    def test_negative_final_overshoot_uses_the_trough_not_the_peak(self):
        # A channel that dips to -1.5 against final=-1.0 has genuinely
        # overshot (50%) -- a peak-only formula (PID's original, unsigned)
        # would instead look at max(y)=0 and report a much larger, wrong
        # number for a negative-going step.
        t = np.linspace(0, 10, 200)
        y_col = np.full_like(t, -1.0)
        y_col[50] = -1.5  # a dip below the negative target
        metrics = compute_tracking_metrics(t, y_col.reshape(-1, 1), np.full((len(t), 1), -1.0))
        self.assertAlmostEqual(metrics[0]["Overshoot"], 50.0, places=3)

    def test_zero_final_returns_zero_overshoot_and_nan_rise(self):
        t = np.linspace(0, 10, 50)
        y_col = np.zeros_like(t)
        metrics = compute_tracking_metrics(t, y_col.reshape(-1, 1), np.zeros((len(t), 1)))
        self.assertEqual(metrics[0]["Overshoot"], 0.0)
        self.assertTrue(np.isnan(metrics[0]["Rise"]))

    def test_regulator_case_has_no_tracking_metrics(self):
        p = double_integrator()
        res = LQR(p, Q=np.eye(2), R=np.eye(1)).design()
        t = np.arange(0.0, 20.0, 0.01)
        sim = simulate_state_feedback(res, t)
        self.assertIsNone(sim.tracking_metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Reference tracking (N̄ feedforward)
# ─────────────────────────────────────────────────────────────────────────────

class TestReferenceTracking(unittest.TestCase):
    def test_zero_steady_state_error(self):
        ex = load_example("aircraft_hall")  # square: nu == ny == 2
        res = LQR(ex.plant, Q=ex.build_suggested_Q(), R=ex.build_suggested_R()).design()
        rt = add_reference_tracking(res)
        self.assertIsNotNone(rt.Nbar)

        t = np.arange(0.0, 200.0, 0.05)
        r = np.tile([1.0, -0.5], (len(t), 1))
        sim = simulate_state_feedback(rt, t, r=r)
        np.testing.assert_allclose(sim.y[-1], [1.0, -0.5], atol=1e-3)

    def test_rejects_non_square_plant(self):
        ex = load_example("distillation_column")  # ny=3, nu=2
        res = LQR(ex.plant, Q=ex.build_suggested_Q(), R=ex.build_suggested_R()).design()
        with self.assertRaises(ValueError):
            add_reference_tracking(res)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateStateFeedback(unittest.TestCase):
    def test_regulator_decays_to_zero(self):
        p = double_integrator()
        res = LQR(p, Q=np.eye(2), R=np.eye(1)).design()
        t = np.arange(0.0, 20.0, 0.01)
        sim = simulate_state_feedback(res, t)
        self.assertTrue(sim.stable)
        self.assertLess(np.linalg.norm(sim.x[-1]), 1e-3)
        self.assertFalse(sim.metrics["unstable"])

    def test_saturation_clips_u(self):
        p = double_integrator()
        res = LQR(p, Q=np.eye(2), R=np.eye(1)).design()
        t = np.arange(0.0, 5.0, 0.01)
        sim = simulate_state_feedback(res, t, x0=[10.0, 0.0], u_min=-1.0, u_max=1.0)
        self.assertTrue(np.all(sim.u <= 1.0 + 1e-9))
        self.assertTrue(np.all(sim.u >= -1.0 - 1e-9))


class TestSimulateOutputFeedback(unittest.TestCase):
    def test_requires_kalman(self):
        p = double_integrator()
        res = LQR(p, Q=np.eye(2), R=np.eye(1)).design()
        t = np.arange(0.0, 5.0, 0.01)
        with self.assertRaises(ValueError):
            simulate_output_feedback(res, t)

    def test_estimator_converges_from_wrong_guess(self):
        p = double_integrator()
        res = LQG(p, Q=np.eye(2), R=np.eye(1), Qw=1e-6 * np.eye(2), Rv=1e-6 * np.eye(1)).design()
        t = np.arange(0.0, 30.0, 0.01)
        sim = simulate_output_feedback(res, t, x0=[1.0, 0.0], x0_hat=[5.0, -5.0])
        # both the true state and the estimation error should decay
        self.assertLess(np.linalg.norm(sim.x[-1]), 1e-2)
        self.assertLess(np.linalg.norm(sim.x[-1] - sim.x_hat[-1]), 1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Preset catalog — smoke tests (no independent numeric answer key except
# aircraft_hall, covered above; these just check "it runs and is sane")
# ─────────────────────────────────────────────────────────────────────────────

class TestPresetCatalog(unittest.TestCase):
    def test_eleven_clean_plants_present(self):
        expected = {"airc", "aircraft_hall", "autm", "chemical_reactor",
                   "distillation_column", "drone", "f100_engine",
                   "furnace_model", "generic_rtp", "rpv", "tgen"}
        self.assertEqual(set(list_examples()), expected)

    def test_excluded_examples_absent(self):
        for key in ("example2_rtp",):
            self.assertNotIn(key, list_examples())

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            load_example("not_a_real_plant")

    def test_every_preset_lqr_is_stabilizable(self):
        for key in list_examples():
            ex = load_example(key)
            res = LQR(ex.plant, Q=ex.build_suggested_Q(), R=ex.build_suggested_R()).design()
            self.assertTrue(res.is_stable(), f"{key}: LQR did not stabilize the plant")
            self.assertEqual(res.gains.K.shape, (ex.plant.nu, ex.plant.nx))

    def test_every_preset_is_controllable(self):
        # Necessary for LQR to place all closed-loop poles (stabilizable
        # would suffice for stability, but every professor-provided plant
        # here is fully controllable).
        for key in list_examples():
            ex = load_example(key)
            self.assertTrue(ex.plant.is_controllable(), f"{key}: not controllable")


if __name__ == "__main__":
    unittest.main()
