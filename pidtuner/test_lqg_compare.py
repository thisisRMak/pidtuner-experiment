"""Unit tests for lqg_compare.py -- the shared comparison core used by both
cli_lqg.py (--method all / model_following_all) and supervisor_tools_lqg.py.

Run with:
    python test_lqg_compare.py
or:
    python -m unittest test_lqg_compare -v
"""

from __future__ import annotations

import unittest

import numpy as np

from lqg_examples import load_example
from lqg_explicit import ExplicitModelFollowingResult
from lqg_compare import compare_regulator_methods, compare_model_following


class TestCompareRegulatorMethods(unittest.TestCase):
    def test_returns_four_rows_in_fixed_order(self):
        ex = load_example("aircraft_hall")
        rows = compare_regulator_methods(ex)
        self.assertEqual([r.name for r in rows],
                         ["LQR (suggested Q/R)", "Output-weighted LQR",
                          "Bryson's rule", "LQG (Kalman filter)"])

    def test_all_rows_share_a_time_axis(self):
        ex = load_example("aircraft_hall")
        rows = compare_regulator_methods(ex)
        t0 = rows[0].sim.t
        for r in rows[1:]:
            np.testing.assert_array_equal(r.sim.t, t0)

    def test_shared_time_axis_covers_the_slowest_method(self):
        # t_end should be at least as long as any individual method's own
        # auto_t_end would pick -- otherwise the plot would truncate the
        # slowest method's settling.
        from lqg_simulate import auto_t_end
        ex = load_example("aircraft_hall")
        rows = compare_regulator_methods(ex)
        shared_t_end = rows[0].sim.t[-1]
        slowest = max(auto_t_end(r.result.closed_loop_poles) for r in rows)
        self.assertGreaterEqual(shared_t_end, slowest)

    def test_all_rows_stable_and_checked(self):
        ex = load_example("aircraft_hall")
        rows = compare_regulator_methods(ex)
        for r in rows:
            self.assertTrue(r.result.is_stable())
            all_checks = r.checks["pre"] + r.checks["post"]
            self.assertTrue(all(c.passed for c in all_checks))

    def test_custom_x_max_u_max_changes_bryson_only(self):
        ex = load_example("chemical_reactor")
        default_rows = compare_regulator_methods(ex)
        custom_rows = compare_regulator_methods(ex, x_max=[1, 2, 3, 4], u_max=[5, 5])
        # Bryson's K should differ...
        self.assertFalse(np.allclose(default_rows[2].result.gains.K, custom_rows[2].result.gains.K))
        # ...but LQR (unaffected by x_max/u_max) should be identical.
        np.testing.assert_allclose(default_rows[0].result.gains.K, custom_rows[0].result.gains.K)

    def test_explicit_t_end_overrides_auto(self):
        ex = load_example("aircraft_hall")
        rows = compare_regulator_methods(ex, t_end=50.0, dt=0.1)
        for r in rows:
            self.assertAlmostEqual(r.sim.t[-1], 50.0, delta=0.11)


class TestCompareRegulatorMethodsCustomWeights(unittest.TestCase):
    def test_Q_diag_R_diag_adds_fifth_row(self):
        ex = load_example("aircraft_hall")
        rows = compare_regulator_methods(ex, Q_diag=[1, 1, 1, 1, 1], R_diag=[0.1, 0.1])
        self.assertEqual([r.name for r in rows],
                         ["LQR (suggested Q/R)", "Output-weighted LQR", "Bryson's rule",
                          "LQG (Kalman filter)", "Custom LQR (Q_diag/R_diag)"])
        self.assertTrue(rows[-1].result.is_stable())

    def test_Q_diag_without_R_diag_rejected(self):
        ex = load_example("aircraft_hall")
        with self.assertRaises(ValueError):
            compare_regulator_methods(ex, Q_diag=[1, 1, 1, 1, 1])

    def test_R_diag_without_Q_diag_rejected(self):
        ex = load_example("aircraft_hall")
        with self.assertRaises(ValueError):
            compare_regulator_methods(ex, R_diag=[0.1, 0.1])

    def test_wrong_length_Q_diag_rejected(self):
        ex = load_example("aircraft_hall")
        with self.assertRaises(ValueError):
            compare_regulator_methods(ex, Q_diag=[1, 1], R_diag=[0.1, 0.1])

    def test_different_weights_produce_different_gains(self):
        ex = load_example("aircraft_hall")
        loose = compare_regulator_methods(ex, Q_diag=[1, 1, 1, 1, 1], R_diag=[10, 10])
        tight = compare_regulator_methods(ex, Q_diag=[1, 1, 1, 1, 1], R_diag=[0.01, 0.01])
        self.assertFalse(np.allclose(loose[-1].result.gains.K, tight[-1].result.gains.K))


class TestCompareRegulatorMethodsReferenceTracking(unittest.TestCase):
    def test_reference_adds_tracking_metrics_to_every_row(self):
        ex = load_example("aircraft_hall")  # square: nu == ny == 2
        rows = compare_regulator_methods(ex, reference=[1.0, -0.5])
        for r in rows:
            self.assertIsNotNone(r.sim.tracking_metrics)
            self.assertEqual(len(r.sim.tracking_metrics), 2)
            for m in r.sim.tracking_metrics:
                self.assertGreaterEqual(m["Overshoot"], 0.0)

    def test_no_reference_means_no_tracking_metrics(self):
        ex = load_example("aircraft_hall")
        rows = compare_regulator_methods(ex)
        for r in rows:
            self.assertIsNone(r.sim.tracking_metrics)

    def test_steady_state_output_matches_reference(self):
        ex = load_example("aircraft_hall")
        rows = compare_regulator_methods(ex, reference=[1.0, -0.5])
        for r in rows:
            np.testing.assert_allclose(r.sim.y[-1], [1.0, -0.5], atol=1e-2)

    def test_non_square_plant_rejected(self):
        ex = load_example("chemical_reactor")  # nu=2, ny=4
        with self.assertRaises(ValueError):
            compare_regulator_methods(ex, reference=[1.0, 1.0, 1.0, 1.0])

    def test_wrong_length_reference_rejected(self):
        ex = load_example("aircraft_hall")
        with self.assertRaises(ValueError):
            compare_regulator_methods(ex, reference=[1.0, -0.5, 0.3])

    def test_custom_weights_change_overshoot(self):
        # The actual point of combining Q_diag/R_diag with reference:
        # different weights should produce measurably different tracking
        # behavior, not just different regulator metrics.
        ex = load_example("aircraft_hall")
        rows = compare_regulator_methods(
            ex, Q_diag=[1, 1, 1, 1, 1], R_diag=[10, 10], reference=[1.0, -0.5])
        custom_row = rows[-1]
        suggested_row = rows[0]
        self.assertNotAlmostEqual(
            custom_row.sim.tracking_metrics[0]["Overshoot"],
            suggested_row.sim.tracking_metrics[0]["Overshoot"],
            places=1,
        )


class TestCompareModelFollowing(unittest.TestCase):
    def setUp(self):
        self.ex = load_example("aircraft_hall")
        self.Am = np.diag([-0.1, -0.07])
        self.Q1 = np.eye(2)
        self.R = np.eye(2)

    def test_returns_implicit_and_explicit_rows(self):
        rows, (t, xm_ref) = compare_model_following(self.ex.plant, self.Am, self.Q1, self.R)
        self.assertEqual([r.name for r in rows],
                         ["Implicit model-following", "Explicit model-following"])
        self.assertIsInstance(rows[1].result, ExplicitModelFollowingResult)

    def test_xm_reference_matches_shared_time_axis(self):
        rows, (t, xm_ref) = compare_model_following(self.ex.plant, self.Am, self.Q1, self.R)
        self.assertEqual(xm_ref.shape, (len(t), 2))
        np.testing.assert_array_equal(rows[0].sim.t, t)
        np.testing.assert_array_equal(rows[1].sim.t, t)

    def test_xm_reference_decays_under_Am(self):
        # Am is stable here, so the target model's own free response should
        # decay toward 0 -- independent of either design's tracking quality.
        rows, (t, xm_ref) = compare_model_following(self.ex.plant, self.Am, self.Q1, self.R)
        np.testing.assert_allclose(xm_ref[-1], [0.0, 0.0], atol=1e-3)

    def test_explicit_tracks_the_model_much_more_tightly_than_implicit(self):
        # The whole point of the comparison: explicit has a live xm(t)
        # feedforward, implicit doesn't -- explicit's final gap from the
        # model should be far smaller.
        rows, (t, xm_ref) = compare_model_following(self.ex.plant, self.Am, self.Q1, self.R)
        implicit_row, explicit_row = rows
        implicit_gap = np.linalg.norm(implicit_row.sim.y[-1] - xm_ref[-1])
        explicit_gap = np.linalg.norm(explicit_row.sim.y[-1] - xm_ref[-1])
        self.assertLess(explicit_gap, implicit_gap)

    def test_both_rows_stable_and_checked(self):
        rows, _ = compare_model_following(self.ex.plant, self.Am, self.Q1, self.R)
        for r in rows:
            self.assertTrue(r.result.is_stable())
            all_checks = r.checks["pre"] + r.checks["post"]
            self.assertTrue(all(c.passed for c in all_checks))

    def test_propagates_unstable_Am_error(self):
        # ExplicitModelFollowing rejects a non-Hurwitz Am at construction
        # (see lqg_explicit.py) -- compare_model_following shouldn't
        # swallow that.
        with self.assertRaises(ValueError):
            compare_model_following(self.ex.plant, np.diag([0.1, -0.07]), self.Q1, self.R)


if __name__ == "__main__":
    unittest.main()
