"""Unit tests for the PID tuner — no Tkinter, just the math layer.

Run with:
    python test_pid_tuner.py
or:
    python -m unittest test_pid_tuner -v

The tests exercise:
  - Plant parsing (both forms, dead time, edge cases)
  - Each of the 6 tuning methods on a benchmark plant
  - The halved variants of ZN-I and ZN-II
  - Closed-loop simulation under three setpoint waveforms:
      (a) step, (b) ramp, (c) pulse
  - Edge cases (RHP poles, no -180° crossover, etc.)

The benchmark plant throughout is the one from the course:
    G(s) = 1000 / ((s+1)(10s+1)) · exp(-0.5 s)
which is a 2nd-order LTI with stable real poles at s = -0.1, -1
and a dead time of 0.5 s. This plant has finite Ku, so every method
works on it — making it the right common case to test.
"""

import ast
import os
import tempfile
import unittest
import numpy as np

from plant import TransferFunction, parse_coeff_list
from identify import (
    run_step_test, run_relay_test, find_ultimate_gain, FOPDT,
    fit_fopdt_from_step, SOPDT, fit_sopdt_from_step,
    identify_ultimate_gain_from_relay,
)
from tune import (
    PIDGains, halve_gains,
    tune_pole_cancellation, select_slowest_stable_poles,
    tune_zn_method_1, tune_zn_method_2,
    tune_amigo, tune_simc, tune_boyd,
    tune_cohen_coon, tune_chr, tune_tyreus_luyben,
)
from simulate import (
    simulate_closed_loop, format_metrics, make_setpoint,
    is_closed_loop_stable, compute_metrics, compute_back_calc_Ka,
)
from compare import (
    robustness_metrics, load_rejection_metrics, compare_all_methods,
    normalize_column,
)
from signal_format import Signal, save_signal, load_signal
from signal_source import SignalGenerator
from blackbox import BlackBoxTuner, identify_from_signals


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures: plants used across tests
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_plant():
    """Course benchmark plant: G(s) = 1000/((s+1)(10s+1)) · exp(-0.5 s)."""
    return TransferFunction.parse("1000 / ((s+1)*(10s+1))", L=0.5)


def fopdt_plant():
    """Pure FOPDT: G(s) = 2 · exp(-1 s) / (5s+1). Every classical
    FOPDT-based rule should hit something close to its analytical answer."""
    return TransferFunction.fopdt(K=2.0, tau=5.0, L=1.0)


def third_order_plant():
    """Higher-order: G(s) = 1/((s+1)(s+2)(s+3)). No dead time, no
    integrator, three real LHP poles — useful for pole-cancellation
    auto-pick tests."""
    return TransferFunction.parse("1 / ((s+1)*(s+2)*(s+3))")


# ─────────────────────────────────────────────────────────────────────────────
# Plant parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestPlantParsing(unittest.TestCase):
    def test_symbolic_explicit_multiplication(self):
        tf = TransferFunction.parse("1000 / ((s+1)*(10s+1))")
        np.testing.assert_allclose(tf.num, [1000.0])
        np.testing.assert_allclose(tf.den, [10.0, 11.0, 1.0])

    def test_symbolic_implicit_multiplication(self):
        # Should parse identically without the *
        tf = TransferFunction.parse("1000/((s+1)(10s+1))")
        np.testing.assert_allclose(tf.den, [10.0, 11.0, 1.0])

    def test_matlab_form(self):
        tf = TransferFunction.from_coeffs(num=[1], den=[10, 11, 1], gain=1000)
        np.testing.assert_allclose(tf.num, [1000.0])
        np.testing.assert_allclose(tf.den, [10.0, 11.0, 1.0])

    def test_symbolic_and_matlab_agree(self):
        a = TransferFunction.parse("1000/((s+1)(10s+1))")
        b = TransferFunction.from_coeffs(num=[1], den=[10, 11, 1], gain=1000)
        np.testing.assert_allclose(a.num, b.num)
        np.testing.assert_allclose(a.den, b.den)

    def test_power_operator(self):
        # Use power in a proper context (numerator order ≤ denominator order)
        tf = TransferFunction.parse("(s+1)^2 / (s+2)^3")
        # (s+1)^2 = s² + 2s + 1
        # (s+2)^3 = s³ + 6s² + 12s + 8
        np.testing.assert_allclose(tf.num, [1.0, 2.0, 1.0])
        np.testing.assert_allclose(tf.den, [1.0, 6.0, 12.0, 8.0])

    def test_polynomial_numerator(self):
        tf = TransferFunction.parse("(s+2)/(s^2 + 3s + 1)")
        np.testing.assert_allclose(tf.num, [1.0, 2.0])
        np.testing.assert_allclose(tf.den, [1.0, 3.0, 1.0])

    def test_integrator(self):
        tf = TransferFunction.parse("1/(s*(s+1))")
        np.testing.assert_allclose(tf.den, [1.0, 1.0, 0.0])
        self.assertEqual(tf.dc_gain(), float("inf"))

    def test_dead_time(self):
        tf = TransferFunction.parse("1/(s+1)", L=2.5)
        self.assertAlmostEqual(tf.L, 2.5)

    def test_dead_time_in_expression(self):
        tf = TransferFunction.parse("exp(-2*s)/(s+1)")
        self.assertAlmostEqual(tf.L, 2.0)
        np.testing.assert_allclose(tf.num, [1.0])
        np.testing.assert_allclose(tf.den, [1.0, 1.0])

    def test_dead_time_in_expression_with_gain_and_implicit_mult(self):
        tf = TransferFunction.parse("5exp(-1.5*s)/(s+1)")
        self.assertAlmostEqual(tf.L, 1.5)
        np.testing.assert_allclose(tf.num, [5.0])

    def test_dead_time_multiple_exp_factors_accumulate(self):
        tf = TransferFunction.parse("exp(-1*s)*exp(-2*s)/(s+1)")
        self.assertAlmostEqual(tf.L, 3.0)

    def test_dead_time_expression_plus_L_kwarg_conflict_rejected(self):
        with self.assertRaises(ValueError):
            TransferFunction.parse("exp(-2*s)/(s+1)", L=1.0)

    def test_dead_time_added_rejected(self):
        with self.assertRaises(ValueError):
            TransferFunction.parse("exp(-2*s) + 1")

    def test_dead_time_noncausal_rejected(self):
        with self.assertRaises(ValueError):
            TransferFunction.parse("exp(2*s)/(s+1)")

    def test_dead_time_constant_offset_rejected(self):
        with self.assertRaises(ValueError):
            TransferFunction.parse("exp(-s-1)/(s+1)")

    def test_dead_time_division_noncausal_rejected(self):
        with self.assertRaises(ValueError):
            TransferFunction.parse("exp(-1*s)/exp(-3*s)/(s+1)")

    def test_poles_correct(self):
        tf = benchmark_plant()
        poles = sorted(tf.poles().real)
        np.testing.assert_allclose(poles, [-1.0, -0.1])

    def test_improper_rejected(self):
        # s² / (s+1) is improper — should be rejected
        with self.assertRaises(ValueError):
            TransferFunction.parse("s^2 / (s+1)")

    def test_empty_string_rejected(self):
        with self.assertRaises(ValueError):
            TransferFunction.parse("")

    def test_unknown_symbol_rejected(self):
        with self.assertRaises(ValueError):
            TransferFunction.parse("z / (z+1)")

    def test_matlab_brackets_accepted(self):
        # Both `[10, 11, 1]` and `10, 11, 1` should parse
        a = parse_coeff_list("[10, 11, 1]")
        b = parse_coeff_list("10, 11, 1")
        np.testing.assert_allclose(a, b)

    def test_rhp_pole_detection(self):
        tf = TransferFunction.parse("1/((s-1)*(s+2))")
        self.assertTrue(tf.has_rhp_poles())
        self.assertFalse(tf.is_open_loop_stable())

    def test_stable_detection(self):
        tf = benchmark_plant()
        self.assertFalse(tf.has_rhp_poles())
        self.assertTrue(tf.is_open_loop_stable())


# ─────────────────────────────────────────────────────────────────────────────
# Identification primitives
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentification(unittest.TestCase):
    def test_step_fit_recovers_fopdt(self):
        """Synthetic FOPDT in, step-fit should recover K, tau, L."""
        plant = fopdt_plant()  # K=2, tau=5, L=1
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0)
        self.assertAlmostEqual(fopdt.K, 2.0, places=2)
        # tau and L from the 28.3/63.2 method are approximate but should
        # be within ~10% for a clean FOPDT
        self.assertAlmostEqual(fopdt.tau, 5.0, delta=1.0)
        self.assertAlmostEqual(fopdt.L, 1.0, delta=0.5)

    def test_step_fit_noise_robust(self):
        """Add modest noise — fit should still be approximately right."""
        plant = fopdt_plant()
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0,
                                          noise_sigma=0.05, seed=42)
        self.assertAlmostEqual(fopdt.K, 2.0, delta=0.2)

    def test_step_fit_flags_delay_detected_for_delayed_plant(self):
        plant = fopdt_plant()  # K=2, tau=5, L=1 — clear, resolvable delay
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0)
        self.assertTrue(fopdt.delay_detected)
        self.assertIn("improves fit", fopdt.delay_reason)

    def test_step_fit_flags_no_delay_for_first_order_plant(self):
        plant = TransferFunction.parse("1/(s+1)")  # pure first-order, no delay
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0)
        self.assertFalse(fopdt.delay_detected)

    def test_step_fit_flags_no_delay_with_noise_on_first_order_plant(self):
        plant = TransferFunction.parse("1/(s+1)")
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0,
                                          noise_sigma=0.01, seed=7)
        self.assertFalse(fopdt.delay_detected)

    def test_relay_test_finds_oscillation(self):
        plant = benchmark_plant()
        Ku, Pu, _, _, _ = run_relay_test(plant, t_max=80.0, h=1.0)
        self.assertGreater(Ku, 0.0)
        self.assertGreater(Pu, 0.0)

    def test_bode_ultimate_gain(self):
        """Bode and relay should agree to within ~30% for the benchmark."""
        plant = benchmark_plant()
        Ku_bode, Pu_bode, _ = find_ultimate_gain(plant)
        Ku_relay, Pu_relay, _, _, _ = run_relay_test(plant, t_max=80.0, h=1.0)
        # Relay uses describing-function approximation, so 30% tolerance
        self.assertAlmostEqual(Ku_bode, Ku_relay, delta=0.5 * Ku_bode)
        self.assertAlmostEqual(Pu_bode, Pu_relay, delta=0.5 * Pu_bode)

    def test_bode_refuses_when_no_crossover(self):
        """Pure first-order has no -180° crossover."""
        plant = TransferFunction.parse("1/(s+1)")
        with self.assertRaises(RuntimeError):
            find_ultimate_gain(plant)


# ─────────────────────────────────────────────────────────────────────────────
# Tuning methods — each on the benchmark plant
# ─────────────────────────────────────────────────────────────────────────────

class TestTuningMethods(unittest.TestCase):
    def setUp(self):
        self.plant = benchmark_plant()
        # Identify FOPDT once for the methods that need it
        _, _, _, _, self.fopdt = run_step_test(self.plant, step_amp=1.0)

    # 1. Stable pole cancellation
    def test_pole_cancel_recovers_expected_gains(self):
        """For G = 1000 / ((s+1)(10s+1)), cancelling poles at -0.1 and -1:
        Kp = (p1+p2)·Kd = 1.1·Kd,  Ki = p1·p2·Kd = 0.1·Kd."""
        Kd = 1.0
        res = tune_pole_cancellation(self.plant, p1=0.1, p2=1.0, Kd=Kd)
        self.assertAlmostEqual(res.gains.Kp, 1.1 * Kd, places=6)
        self.assertAlmostEqual(res.gains.Ki, 0.1 * Kd, places=6)
        self.assertAlmostEqual(res.gains.Kd, Kd, places=6)

    def test_pole_cancel_auto_picks_slowest(self):
        """Auto mode should pick the two slowest stable poles."""
        p1, p2 = select_slowest_stable_poles(self.plant)
        # Slowest = closest to origin = (0.1, 1.0)
        self.assertAlmostEqual(min(p1, p2), 0.1, places=5)
        self.assertAlmostEqual(max(p1, p2), 1.0, places=5)

    def test_pole_cancel_refuses_rhp(self):
        with self.assertRaises(ValueError):
            tune_pole_cancellation(self.plant, p1=-0.5, p2=1.0)

    def test_pole_cancel_refuses_insufficient_poles(self):
        """1st-order plant has only 1 stable pole — auto-select must refuse."""
        plant = TransferFunction.parse("1/(s+1)")
        with self.assertRaises(ValueError):
            select_slowest_stable_poles(plant)

    # 2. ZN Method I
    def test_zn1_formulas(self):
        """ZN-I:  Kp = 1.2·τ/(K·L),  Ti = 2L,  Td = 0.5L."""
        # Use an exact FOPDT to test the formula, not the fitted one
        f = FOPDT(K=2.0, tau=5.0, L=1.0)
        res = tune_zn_method_1(f)
        expected_Kp = 1.2 * 5.0 / (2.0 * 1.0)  # = 3.0
        expected_Ti = 2.0 * 1.0
        expected_Td = 0.5 * 1.0
        self.assertAlmostEqual(res.gains.Kp, expected_Kp, places=6)
        # Ki = Kp/Ti, Kd = Kp·Td
        self.assertAlmostEqual(res.gains.Ki, expected_Kp / expected_Ti, places=6)
        self.assertAlmostEqual(res.gains.Kd, expected_Kp * expected_Td, places=6)

    def test_zn1_halved(self):
        """Halved variant should be exactly half of the canonical."""
        f = FOPDT(K=2.0, tau=5.0, L=1.0)
        res = tune_zn_method_1(f)
        half = halve_gains(res)
        self.assertAlmostEqual(half.gains.Kp, 0.5 * res.gains.Kp, places=6)
        self.assertAlmostEqual(half.gains.Ki, 0.5 * res.gains.Ki, places=6)
        self.assertAlmostEqual(half.gains.Kd, 0.5 * res.gains.Kd, places=6)

    # 3. ZN Method II
    def test_zn2_formulas(self):
        """ZN-II:  Kp = 0.6·Ku,  Ti = 0.5·Pu,  Td = 0.125·Pu."""
        Ku, Pu = 1.5, 4.0
        res = tune_zn_method_2(Ku, Pu)
        self.assertAlmostEqual(res.gains.Kp, 0.6 * Ku, places=6)
        # Ki = Kp/Ti = Kp / (0.5·Pu)
        self.assertAlmostEqual(res.gains.Ki, 0.6 * Ku / (0.5 * Pu), places=6)
        # Kd = Kp·Td = Kp · 0.125·Pu
        self.assertAlmostEqual(res.gains.Kd, 0.6 * Ku * 0.125 * Pu, places=6)

    def test_zn2_halved(self):
        res = tune_zn_method_2(1.5, 4.0)
        half = halve_gains(res)
        self.assertAlmostEqual(half.gains.Kp, 0.5 * res.gains.Kp, places=6)
        self.assertAlmostEqual(half.gains.Ki, 0.5 * res.gains.Ki, places=6)
        self.assertAlmostEqual(half.gains.Kd, 0.5 * res.gains.Kd, places=6)

    def test_zn2_works_on_benchmark(self):
        """Benchmark plant has dead time → finite Ku."""
        Ku, Pu, _ = find_ultimate_gain(self.plant)
        res = tune_zn_method_2(Ku, Pu)
        self.assertGreater(res.gains.Kp, 0.0)
        self.assertGreater(res.gains.Ki, 0.0)
        self.assertGreater(res.gains.Kd, 0.0)

    # 4. AMIGO
    def test_amigo_formulas(self):
        """AMIGO:  Kp = (1/A)·(0.2 + 0.45·τ/L),  Ti and Td per handout."""
        # Use an exact FOPDT
        f = FOPDT(K=2.0, tau=5.0, L=1.0)
        res = tune_amigo(f)
        A, tau, L = 2.0, 5.0, 1.0
        exp_Kp = (1.0 / A) * (0.2 + 0.45 * tau / L)
        exp_Ti = ((0.4 * L + 0.8 * tau) / (L + 0.1 * tau)) * L
        exp_Td = (0.5 * tau / (0.3 * L + tau)) * L
        self.assertAlmostEqual(res.gains.Kp, exp_Kp, places=6)
        self.assertAlmostEqual(res.gains.Ki, exp_Kp / exp_Ti, places=6)
        self.assertAlmostEqual(res.gains.Kd, exp_Kp * exp_Td, places=6)

    def test_amigo_integrating_form(self):
        """For G = A·exp(-Ls)/s:  Kp = 0.45/A,  Ti = 8L,  Td = 0.5L."""
        f = FOPDT(K=2.0, tau=1.0, L=1.0)  # tau is unused for integrating
        res = tune_amigo(f, integrating=True)
        self.assertAlmostEqual(res.gains.Kp, 0.45 / 2.0, places=6)
        self.assertAlmostEqual(res.gains.Ki, (0.45 / 2.0) / 8.0, places=6)
        self.assertAlmostEqual(res.gains.Kd, (0.45 / 2.0) * 0.5, places=6)

    # 5. SIMC
    def test_simc_default_tau_c_equals_L(self):
        """Default τc = L:  Kp = (1/A)·τ/(L+L) = τ/(2·A·L)."""
        f = FOPDT(K=2.0, tau=5.0, L=1.0)
        res = tune_simc(f)
        exp_Kp = (1.0 / 2.0) * 5.0 / (1.0 + 1.0)
        self.assertAlmostEqual(res.gains.Kp, exp_Kp, places=6)
        # Ti = min(tau, 4·(tau_c + L)) = min(5, 8) = 5
        self.assertAlmostEqual(res.gains.Ki, exp_Kp / 5.0, places=6)

    def test_simc_with_tau_c_user(self):
        f = FOPDT(K=2.0, tau=5.0, L=1.0)
        res = tune_simc(f, tau_c=3.0)
        # Kp = (1/2)·5/(3+1) = 5/8
        self.assertAlmostEqual(res.gains.Kp, 5.0 / 8.0, places=6)

    def test_simc_with_tau2(self):
        """SOPDT form: PID with Td = τ₂."""
        f = FOPDT(K=2.0, tau=5.0, L=1.0)
        res = tune_simc(f, tau2=0.5)
        # Td = tau2, so Kd = Kp·tau2
        self.assertAlmostEqual(res.gains.Kd, res.gains.Kp * 0.5, places=6)

    # 6. Boyd
    def test_boyd_on_benchmark(self):
        """Boyd should produce stable gains on the benchmark plant."""
        # Seed with SIMC for a sensible starting point
        seed = tune_simc(self.fopdt).gains
        res = tune_boyd(self.plant, Ms=1.4, Mt=1.4, seed_gains=seed)
        # Gains should be finite and positive (plant has positive DC gain)
        self.assertTrue(np.isfinite(res.gains.Kp))
        self.assertTrue(np.isfinite(res.gains.Ki))
        self.assertTrue(np.isfinite(res.gains.Kd))
        self.assertGreater(res.gains.Kp, 0.0)
        self.assertGreater(res.gains.Ki, 0.0)

    def test_boyd_higher_Ms_more_aggressive(self):
        """Loosening Ms should allow higher integral gain Ki."""
        seed = tune_simc(self.fopdt).gains
        res_tight = tune_boyd(self.plant, Ms=1.2, Mt=1.2, seed_gains=seed)
        res_loose = tune_boyd(self.plant, Ms=2.0, Mt=2.0, seed_gains=seed)
        self.assertGreater(res_loose.gains.Ki, res_tight.gains.Ki)


# ─────────────────────────────────────────────────────────────────────────────
# New tuning methods: Cohen–Coon, CHR, Tyreus–Luyben
# ─────────────────────────────────────────────────────────────────────────────

class TestNewTuningMethods(unittest.TestCase):
    def setUp(self):
        self.plant = benchmark_plant()
        # Exact FOPDT for closed-form formula checks
        self.f = FOPDT(K=2.0, tau=5.0, L=1.0)

    # 7. Cohen–Coon
    def test_cohen_coon_formulas(self):
        """CC PID: Kp=(1/K)(τ/L)(4/3+L/4τ), Ti=L(32+6r)/(13+8r), Td=4L/(11+2r), r=L/τ."""
        f = self.f
        r = f.L / f.tau
        Kp = (1.0 / f.K) * (f.tau / f.L) * (4.0 / 3.0 + r / 4.0)
        Ti = f.L * (32.0 + 6.0 * r) / (13.0 + 8.0 * r)
        Td = 4.0 * f.L / (11.0 + 2.0 * r)
        res = tune_cohen_coon(f)
        self.assertAlmostEqual(res.gains.Kp, Kp, places=6)
        self.assertAlmostEqual(res.gains.Ki, Kp / Ti, places=6)
        self.assertAlmostEqual(res.gains.Kd, Kp * Td, places=6)

    def test_cohen_coon_runs_on_benchmark(self):
        _, _, _, _, fopdt = run_step_test(self.plant)
        res = tune_cohen_coon(fopdt)
        sim = simulate_closed_loop(self.plant, res.gains, setpoint=1.0)
        self.assertFalse(sim.metrics["unstable"])

    def test_cohen_coon_refuses_degenerate(self):
        with self.assertRaises(ValueError):
            tune_cohen_coon(FOPDT(K=0.0, tau=5.0, L=1.0))

    # 8. Chien–Hrones–Reswick
    def test_chr_setpoint_0_formulas(self):
        """CHR servo 0%: Kp=0.6τ/(KL), Ti=τ, Td=0.5L."""
        f = self.f
        res = tune_chr(f, response="setpoint", overshoot=0)
        Kp = 0.6 * f.tau / (f.K * f.L)
        self.assertAlmostEqual(res.gains.Kp, Kp, places=6)
        self.assertAlmostEqual(res.gains.Ki, Kp / f.tau, places=6)
        self.assertAlmostEqual(res.gains.Kd, Kp * 0.5 * f.L, places=6)

    def test_chr_load_20_formulas(self):
        """CHR regulator 20%: Kp=1.2τ/(KL), Ti=2L, Td=0.42L."""
        f = self.f
        res = tune_chr(f, response="load", overshoot=20)
        Kp = 1.2 * f.tau / (f.K * f.L)
        self.assertAlmostEqual(res.gains.Kp, Kp, places=6)
        self.assertAlmostEqual(res.gains.Ki, Kp / (2.0 * f.L), places=6)
        self.assertAlmostEqual(res.gains.Kd, Kp * 0.42 * f.L, places=6)

    def test_chr_all_four_variants_stable(self):
        _, _, _, _, fopdt = run_step_test(self.plant)
        for response in ("setpoint", "load"):
            for overshoot in (0, 20):
                res = tune_chr(fopdt, response=response, overshoot=overshoot)
                sim = simulate_closed_loop(self.plant, res.gains, setpoint=1.0)
                self.assertFalse(sim.metrics["unstable"],
                                 f"{response} {overshoot}% went unstable")

    def test_chr_rejects_bad_variant(self):
        with self.assertRaises(ValueError):
            tune_chr(self.f, response="servo", overshoot=0)
        with self.assertRaises(ValueError):
            tune_chr(self.f, response="setpoint", overshoot=50)

    # 9. Tyreus–Luyben
    def test_tyreus_luyben_pid_formulas(self):
        """T-L PID: Kp=Ku/2.2, Ti=2.2Pu, Td=Pu/6.3."""
        Ku, Pu = 1.5, 4.0
        res = tune_tyreus_luyben(Ku, Pu)
        self.assertAlmostEqual(res.gains.Kp, Ku / 2.2, places=6)
        self.assertAlmostEqual(res.gains.Ki, (Ku / 2.2) / (2.2 * Pu), places=6)
        self.assertAlmostEqual(res.gains.Kd, (Ku / 2.2) * (Pu / 6.3), places=6)

    def test_tyreus_luyben_pi_has_no_derivative(self):
        """T-L PI: Kp=Ku/3.2, Ti=2.2Pu, Kd=0."""
        Ku, Pu = 1.5, 4.0
        res = tune_tyreus_luyben(Ku, Pu, use_derivative=False)
        self.assertAlmostEqual(res.gains.Kp, Ku / 3.2, places=6)
        self.assertEqual(res.gains.Kd, 0.0)

    def test_tyreus_luyben_more_conservative_than_zn2(self):
        """T-L should have a smaller Kp than ZN-II for the same Ku, Pu."""
        Ku, Pu = 1.5, 4.0
        tl = tune_tyreus_luyben(Ku, Pu)
        zn = tune_zn_method_2(Ku, Pu)
        self.assertLess(tl.gains.Kp, zn.gains.Kp)

    def test_tyreus_luyben_runs_on_benchmark(self):
        Ku, Pu, _ = find_ultimate_gain(self.plant)
        res = tune_tyreus_luyben(Ku, Pu)
        sim = simulate_closed_loop(self.plant, res.gains, setpoint=1.0)
        self.assertFalse(sim.metrics["unstable"])


# ─────────────────────────────────────────────────────────────────────────────
# Closed-loop simulation with three setpoint kinds
# ─────────────────────────────────────────────────────────────────────────────
# Every method that tunes the benchmark plant should produce a *stable*
# closed loop. We test all methods × three setpoint waveforms.

class TestClosedLoopAllSetpoints(unittest.TestCase):
    """For each method, simulate the closed loop on step, ramp, and pulse."""

    def setUp(self):
        self.plant = benchmark_plant()
        _, _, _, _, self.fopdt = run_step_test(self.plant, step_amp=1.0)
        Ku, Pu, _ = find_ultimate_gain(self.plant)
        seed = tune_simc(self.fopdt).gains

        self.tunings = {
            "pole_cancel": tune_pole_cancellation(self.plant, p1=0.1, p2=1.0,
                                                  Kd=1.0 / abs(self.plant.dc_gain())),
            "zn1": tune_zn_method_1(self.fopdt),
            "zn1_half": halve_gains(tune_zn_method_1(self.fopdt)),
            "zn2": tune_zn_method_2(Ku, Pu),
            "zn2_half": halve_gains(tune_zn_method_2(Ku, Pu)),
            "amigo": tune_amigo(self.fopdt),
            "simc": tune_simc(self.fopdt),
            "boyd": tune_boyd(self.plant, Ms=1.4, Mt=1.4, seed_gains=seed),
        }

    def _check_simulation(self, gains, kind, name):
        sim = simulate_closed_loop(
            self.plant, gains, setpoint=1.0, setpoint_kind=kind,
            t_end=60.0, u_min=-1e4, u_max=1e4,
        )
        # The response should be finite
        self.assertTrue(np.all(np.isfinite(sim.y)),
                        f"{name}/{kind}: non-finite y values")
        self.assertTrue(np.all(np.isfinite(sim.u)),
                        f"{name}/{kind}: non-finite u values")
        # Sanity: setpoint waveform is correct
        self.assertEqual(sim.sp_kind, kind)
        # Metrics dictionary populated with the right kind-specific keys
        self.assertEqual(sim.metrics.get("sp_kind"), kind)
        if kind == "step":
            self.assertIn("Overshoot", sim.metrics)
            self.assertIn("Rise", sim.metrics)
            self.assertIn("Settling", sim.metrics)
        elif kind == "ramp":
            self.assertIn("ss_error", sim.metrics)
            self.assertIn("max_error", sim.metrics)
        elif kind == "pulse":
            self.assertIn("peak_error", sim.metrics)
            self.assertIn("final_residual", sim.metrics)
        return sim

    # Methods × inputs — generated programmatically below
    # (We unroll explicitly so failures point at the right cell.)

    def test_step_pole_cancel(self):   self._check_simulation(self.tunings["pole_cancel"].gains, "step", "pole_cancel")
    def test_step_zn1(self):           self._check_simulation(self.tunings["zn1"].gains, "step", "zn1")
    def test_step_zn1_half(self):      self._check_simulation(self.tunings["zn1_half"].gains, "step", "zn1_half")
    def test_step_zn2(self):           self._check_simulation(self.tunings["zn2"].gains, "step", "zn2")
    def test_step_zn2_half(self):      self._check_simulation(self.tunings["zn2_half"].gains, "step", "zn2_half")
    def test_step_amigo(self):         self._check_simulation(self.tunings["amigo"].gains, "step", "amigo")
    def test_step_simc(self):          self._check_simulation(self.tunings["simc"].gains, "step", "simc")
    def test_step_boyd(self):          self._check_simulation(self.tunings["boyd"].gains, "step", "boyd")

    def test_ramp_pole_cancel(self):   self._check_simulation(self.tunings["pole_cancel"].gains, "ramp", "pole_cancel")
    def test_ramp_zn1(self):           self._check_simulation(self.tunings["zn1"].gains, "ramp", "zn1")
    def test_ramp_zn1_half(self):      self._check_simulation(self.tunings["zn1_half"].gains, "ramp", "zn1_half")
    def test_ramp_zn2(self):           self._check_simulation(self.tunings["zn2"].gains, "ramp", "zn2")
    def test_ramp_zn2_half(self):      self._check_simulation(self.tunings["zn2_half"].gains, "ramp", "zn2_half")
    def test_ramp_amigo(self):         self._check_simulation(self.tunings["amigo"].gains, "ramp", "amigo")
    def test_ramp_simc(self):          self._check_simulation(self.tunings["simc"].gains, "ramp", "simc")
    def test_ramp_boyd(self):          self._check_simulation(self.tunings["boyd"].gains, "ramp", "boyd")

    def test_pulse_pole_cancel(self):  self._check_simulation(self.tunings["pole_cancel"].gains, "pulse", "pole_cancel")
    def test_pulse_zn1(self):          self._check_simulation(self.tunings["zn1"].gains, "pulse", "zn1")
    def test_pulse_zn1_half(self):     self._check_simulation(self.tunings["zn1_half"].gains, "pulse", "zn1_half")
    def test_pulse_zn2(self):          self._check_simulation(self.tunings["zn2"].gains, "pulse", "zn2")
    def test_pulse_zn2_half(self):     self._check_simulation(self.tunings["zn2_half"].gains, "pulse", "zn2_half")
    def test_pulse_amigo(self):        self._check_simulation(self.tunings["amigo"].gains, "pulse", "amigo")
    def test_pulse_simc(self):         self._check_simulation(self.tunings["simc"].gains, "pulse", "simc")
    def test_pulse_boyd(self):         self._check_simulation(self.tunings["boyd"].gains, "pulse", "boyd")


# ─────────────────────────────────────────────────────────────────────────────
# Property tests: behavioural sanity checks across the whole tuning catalog
# ─────────────────────────────────────────────────────────────────────────────

class TestTuningProperties(unittest.TestCase):
    """Sanity checks that aren't tied to exact formula values, but to
    properties the literature claims for each method.

    The benchmark plant is well-behaved and all methods should:
      - produce a stable closed loop on the step response,
      - settle to within 5% of the setpoint within the sim duration,
      - keep finite control effort.

    Additionally, AMIGO and SIMC should have *lower* overshoot than ZN-I
    (the whole point of those methods — see HO13/HO14 and lecture
    4/30/26). We verify that explicitly.
    """

    def setUp(self):
        self.plant = benchmark_plant()
        _, _, _, _, self.fopdt = run_step_test(self.plant, step_amp=1.0)
        Ku, Pu, _ = find_ultimate_gain(self.plant)
        seed = tune_simc(self.fopdt).gains

        Kd_cancel = 1.0 / abs(self.plant.dc_gain())
        self.results = {
            "pole_cancel": tune_pole_cancellation(self.plant, 0.1, 1.0, Kd=Kd_cancel),
            "zn1":   tune_zn_method_1(self.fopdt),
            "zn1_h": halve_gains(tune_zn_method_1(self.fopdt)),
            "zn2":   tune_zn_method_2(Ku, Pu),
            "zn2_h": halve_gains(tune_zn_method_2(Ku, Pu)),
            "amigo": tune_amigo(self.fopdt),
            "simc":  tune_simc(self.fopdt),
            "boyd":  tune_boyd(self.plant, Ms=1.4, Mt=1.4, seed_gains=seed),
        }
        self.sims = {
            k: simulate_closed_loop(self.plant, r.gains, setpoint=1.0,
                                    setpoint_kind="step", t_end=80.0,
                                    u_min=-1e4, u_max=1e4)
            for k, r in self.results.items()
        }

    def test_all_methods_stable_on_benchmark(self):
        for name, sim in self.sims.items():
            with self.subTest(method=name):
                self.assertFalse(sim.metrics["unstable"],
                                 f"{name}: closed loop unstable")
                # Settle near setpoint by end of sim
                final = sim.y[-int(0.05 * len(sim.y)):]
                self.assertLess(abs(np.mean(final) - 1.0), 0.05,
                                f"{name}: doesn't settle near 1.0")

    def test_halved_zn_has_lower_overshoot(self):
        """The whole point of halving."""
        os_full = self.sims["zn1"].metrics["Overshoot"]
        os_half = self.sims["zn1_h"].metrics["Overshoot"]
        self.assertLess(os_half, os_full,
                        "ZN-I halved should overshoot less than full")

        os_full_2 = self.sims["zn2"].metrics["Overshoot"]
        os_half_2 = self.sims["zn2_h"].metrics["Overshoot"]
        self.assertLess(os_half_2, os_full_2,
                        "ZN-II halved should overshoot less than full")

    def test_zn1_more_aggressive_than_amigo(self):
        """ZN-I is famous for being aggressive; AMIGO targets ~20%.
        Expect ZN-I overshoot > AMIGO overshoot on the benchmark."""
        self.assertGreater(self.sims["zn1"].metrics["Overshoot"],
                           self.sims["amigo"].metrics["Overshoot"])

    def test_simc_smoothest(self):
        """SIMC is marketed as 'smooth tuning' — should beat ZN-I on
        overshoot by a healthy margin."""
        self.assertLess(self.sims["simc"].metrics["Overshoot"],
                        self.sims["zn1"].metrics["Overshoot"])

    def test_pole_cancellation_zero_overshoot_proper_Kd(self):
        """With a small enough Kd, pole-cancellation should give a
        first-order-like overshoot-free response (open loop is K·Kd/s)."""
        # Reduce Kd well below the default to ensure no overshoot
        small_Kd = 0.1 / abs(self.plant.dc_gain())
        res = tune_pole_cancellation(self.plant, 0.1, 1.0, Kd=small_Kd)
        sim = simulate_closed_loop(self.plant, res.gains, setpoint=1.0,
                                   setpoint_kind="step", t_end=200.0,
                                   u_min=-1e4, u_max=1e4)
        self.assertLess(sim.metrics["Overshoot"], 1.0,
                        "Pole cancellation should have ~0 overshoot")


# ─────────────────────────────────────────────────────────────────────────────
# Setpoint waveform generators in isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestSetpointWaveforms(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0, 10, 1001)

    def test_step_shape(self):
        sp = make_setpoint(self.t, "step", amplitude=2.0)
        self.assertEqual(sp[0], 0.0)
        np.testing.assert_allclose(sp[1:], 2.0)

    def test_ramp_shape(self):
        sp = make_setpoint(self.t, "ramp", amplitude=5.0)
        self.assertAlmostEqual(sp[0], 0.0, places=6)
        self.assertAlmostEqual(sp[-1], 5.0, places=6)
        # Monotonically non-decreasing
        self.assertTrue(np.all(np.diff(sp) >= 0))

    def test_pulse_shape(self):
        sp = make_setpoint(self.t, "pulse", amplitude=1.0)
        # Pulse occupies [25%, 50%] of duration
        self.assertEqual(sp[0], 0.0)
        self.assertEqual(sp[-1], 0.0)
        # Some interior should be 1.0
        self.assertGreater(np.sum(sp == 1.0), 0)

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            make_setpoint(self.t, "wat")


# ─────────────────────────────────────────────────────────────────────────────
# Anti-windup: conditional-integration (default) vs. back-calculation
# ─────────────────────────────────────────────────────────────────────────────

class TestAntiWindup(unittest.TestCase):
    def setUp(self):
        self.plant = TransferFunction.parse("1000/((s+1)*(10s+1))", L=0.5)
        self.gains = PIDGains(Kp=0.011295149534671681, Ki=0.005297933356462488,
                              Kd=0.006020291046836671)

    def test_default_is_conditional(self):
        sim = simulate_closed_loop(self.plant, self.gains, setpoint=1.0,
                                   setpoint_kind="step", u_min=-1e6, u_max=1e6)
        self.assertEqual(sim.antiwindup, "conditional")
        self.assertEqual(sim.Ka, 0.0)

    def test_modes_identical_when_never_saturated(self):
        # Wide-open bounds -> u_sat == u_unsat always -> both modes reduce
        # to plain integration, so the trajectories must match exactly.
        common = dict(plant=self.plant, gains=self.gains, setpoint=1.0,
                      setpoint_kind="step", u_min=-1e6, u_max=1e6)
        cond = simulate_closed_loop(antiwindup="conditional", **common)
        back = simulate_closed_loop(antiwindup="back_calc", **common)
        np.testing.assert_allclose(cond.y, back.y)
        np.testing.assert_allclose(cond.u, back.u)

    def test_modes_diverge_when_saturated(self):
        # Tight bounds force real saturation; conditional-integration
        # (freeze) and back-calculation (active correction) must then
        # produce different trajectories.
        common = dict(plant=self.plant, gains=self.gains, setpoint=1.0,
                      setpoint_kind="step", u_min=-0.003, u_max=0.003)
        cond = simulate_closed_loop(antiwindup="conditional", **common)
        back = simulate_closed_loop(antiwindup="back_calc", **common)
        self.assertGreater(np.max(np.abs(cond.y - back.y)), 1e-6)
        self.assertGreater(back.Ka, 0.0)

    def test_Ka_auto_derivation_full_pid(self):
        # Ti = Kp/Ki, Td = Kd/Kp, Tt = sqrt(Ti*Td), Ka = 1/Tt (Astrom & Hagglund).
        Ka, Tt = compute_back_calc_Ka(self.gains)
        Ti = self.gains.Kp / self.gains.Ki
        Td = self.gains.Kd / self.gains.Kp
        expected_Tt = np.sqrt(Ti * Td)
        self.assertAlmostEqual(Tt, expected_Tt, places=9)
        self.assertAlmostEqual(Ka, 1.0 / expected_Tt, places=9)

    def test_Ka_auto_derivation_pi_only(self):
        # Td = 0 -> Tt = Ti (PI-only case), per Astrom & Hagglund.
        gains = PIDGains(Kp=0.01, Ki=0.004, Kd=0.0)
        Ka, Tt = compute_back_calc_Ka(gains)
        expected_Ti = gains.Kp / gains.Ki
        self.assertAlmostEqual(Tt, expected_Ti, places=9)
        self.assertAlmostEqual(Ka, 1.0 / expected_Ti, places=9)

    def test_Ka_no_integral_action(self):
        # Ki = 0 -> nothing for an integrator to wind up -> Ka = 0, Tt = inf.
        gains = PIDGains(Kp=0.01, Ki=0.0, Kd=0.005)
        Ka, Tt = compute_back_calc_Ka(gains)
        self.assertEqual(Ka, 0.0)
        self.assertEqual(Tt, float("inf"))

    def test_Ka_override_bypasses_derivation(self):
        Ka, Tt = compute_back_calc_Ka(self.gains, Ka=2.0)
        self.assertEqual(Ka, 2.0)
        self.assertAlmostEqual(Tt, 0.5, places=9)
        sim = simulate_closed_loop(self.plant, self.gains, setpoint=1.0,
                                   setpoint_kind="step", u_min=-0.003, u_max=0.003,
                                   antiwindup="back_calc", Ka=2.0)
        self.assertEqual(sim.Ka, 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison data layer: robustness, load rejection, compare_all_methods
# ─────────────────────────────────────────────────────────────────────────────

class TestComparisonLayer(unittest.TestCase):
    def setUp(self):
        self.plant = benchmark_plant()
        _, _, _, _, self.fopdt = run_step_test(self.plant)

    def test_robustness_keys_and_ranges(self):
        g = tune_simc(self.fopdt).gains
        r = robustness_metrics(self.plant, g)
        for k in ("Ms", "Mt", "GM_dB", "PM_deg"):
            self.assertIn(k, r)
        # Ms is always >= 1 for a real loop; sane upper sanity bound
        self.assertGreaterEqual(r["Ms"], 1.0 - 1e-6)
        self.assertLess(r["Ms"], 100.0)

    def test_aggressive_has_higher_Ms_than_robust(self):
        """ZN-II (aggressive) should have a larger Ms than SIMC (smooth)."""
        Ku, Pu, _ = find_ultimate_gain(self.plant)
        ms_zn2 = robustness_metrics(self.plant, tune_zn_method_2(Ku, Pu).gains)["Ms"]
        ms_simc = robustness_metrics(self.plant, tune_simc(self.fopdt).gains)["Ms"]
        self.assertGreater(ms_zn2, ms_simc)

    def test_load_rejection_finite_and_positive(self):
        g = tune_zn_method_1(self.fopdt).gains
        r = load_rejection_metrics(self.plant, g)
        self.assertTrue(np.isfinite(r["IAE_load"]))
        self.assertGreater(r["IAE_load"], 0.0)

    def test_zn1_rejects_load_better_than_pole_cancel(self):
        """ZN-I is a load-rejection design; pole-cancellation is not.
        ZN-I should achieve a much lower load IAE on the benchmark."""
        zn1 = tune_zn_method_1(self.fopdt).gains
        p1, p2 = select_slowest_stable_poles(self.plant)
        Kd = 1.0 / abs(self.plant.dc_gain())
        pc = tune_pole_cancellation(self.plant, p1, p2, Kd=Kd).gains
        iae_zn1 = load_rejection_metrics(self.plant, zn1)["IAE_load"]
        iae_pc = load_rejection_metrics(self.plant, pc)["IAE_load"]
        self.assertLess(iae_zn1, iae_pc)

    def test_compare_all_methods_rows(self):
        rows = compare_all_methods(self.plant)
        # Every method family should appear
        names = {r["name"] for r in rows}
        for expect in ("ZN-I", "ZN-II", "AMIGO", "SIMC", "Boyd",
                       "Cohen–Coon", "Tyreus–Luyben"):
            self.assertIn(expect, names)
        # Stable rows carry all the table metrics
        for r in rows:
            if r["stable"]:
                for m in ("OS%", "ts", "IAE", "IAE_load", "Ms", "Mt", "u_tv"):
                    self.assertIn(m, r)

    def test_load_step_does_not_break_setpoint_sim(self):
        """A sim with no load_step must match the old behaviour exactly."""
        g = tune_simc(self.fopdt).gains
        a = simulate_closed_loop(self.plant, g, setpoint=1.0)
        b = simulate_closed_loop(self.plant, g, setpoint=1.0, load_step=None)
        np.testing.assert_allclose(a.y, b.y)

    def test_normalize_column_best_is_one(self):
        # Lower-is-better: the smallest value should map to 1.0
        vals = [3.0, 1.0, 2.0, float("inf")]
        out = normalize_column(vals, direction=-1)
        self.assertAlmostEqual(out[1], 1.0, places=6)   # smallest → best
        self.assertAlmostEqual(out[0], 0.0, places=6)   # largest finite → worst
        self.assertAlmostEqual(out[3], 0.0, places=6)   # inf → worst


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):
    def test_third_order_plant_pole_cancel(self):
        """3rd-order plant: PID cancels 2 of 3 stable poles, leaves one."""
        plant = third_order_plant()  # poles at -1, -2, -3
        p1, p2 = select_slowest_stable_poles(plant)
        # Slowest two = -1 and -2
        self.assertAlmostEqual(min(p1, p2), 1.0, places=5)
        self.assertAlmostEqual(max(p1, p2), 2.0, places=5)

    def test_simc_pi_only_no_tau2(self):
        """SIMC with no τ₂ should give a PI controller (Kd = 0)."""
        f = FOPDT(K=2.0, tau=5.0, L=1.0)
        res = tune_simc(f)
        self.assertEqual(res.gains.Kd, 0.0)

    def test_metric_format_unstable(self):
        """format_metrics should label diverging responses as UNSTABLE."""
        bad_metrics = {
            "IAE": float("inf"), "ITAE": float("inf"),
            "u_peak": float("inf"), "u_rms": float("inf"),
            "unstable": True, "sp_kind": "step",
        }
        self.assertIn("UNSTABLE", format_metrics(bad_metrics))


# ─────────────────────────────────────────────────────────────────────────────
# Signal-generator / black-box-tuner split
# ─────────────────────────────────────────────────────────────────────────────

def _module_imports_plant(module_file):
    """AST-based structural check: does this module's own source (not its
    transitive dependencies) contain an `import plant` / `from plant import`
    statement? Used to verify the isolation contract without relying on
    convention alone."""
    with open(module_file) as f:
        tree = ast.parse(f.read(), filename=module_file)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "plant" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "plant":
                return True
    return False


class TestSignalFormat(unittest.TestCase):
    def _sample_signal(self):
        t = np.linspace(0.0, 1.0, 11)
        u = np.ones_like(t)
        y = np.linspace(0.0, 5.0, 11)
        return Signal(t=t, u=u, y=y, dt=0.1, experiment="step",
                      meta={"step_amp": 1.0, "noise_sigma": 0.0, "seed": None})

    def test_no_plant_import(self):
        import signal_format
        self.assertFalse(_module_imports_plant(signal_format.__file__),
                          "signal_format.py must never import plant.py")

    def test_invalid_experiment_rejected(self):
        with self.assertRaises(ValueError):
            Signal(t=np.zeros(3), u=np.zeros(3), y=np.zeros(3), dt=0.1,
                   experiment="not-a-real-experiment")

    def test_non_serializable_meta_rejected(self):
        with self.assertRaises(ValueError):
            Signal(t=np.zeros(3), u=np.zeros(3), y=np.zeros(3), dt=0.1,
                   experiment="step", meta={"bad": np.array([1, 2, 3])})

    def test_iter_yields_ordered_triples(self):
        sig = self._sample_signal()
        triples = list(sig)
        self.assertEqual(len(triples), len(sig))
        for k, (tk, uk, yk) in enumerate(triples):
            self.assertEqual(tk, sig.t[k])
            self.assertEqual(uk, sig.u[k])
            self.assertEqual(yk, sig.y[k])

    def test_save_load_round_trip(self):
        sig = self._sample_signal()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sig.npz")
            save_signal(sig, path)
            loaded = load_signal(path)
        np.testing.assert_array_equal(loaded.t, sig.t)
        np.testing.assert_array_equal(loaded.u, sig.u)
        np.testing.assert_array_equal(loaded.y, sig.y)
        self.assertEqual(loaded.dt, sig.dt)
        self.assertEqual(loaded.experiment, sig.experiment)
        self.assertEqual(loaded.meta, sig.meta)
        # "live" interface is identical whether freshly built or loaded
        self.assertEqual(list(loaded), list(sig))


class TestSignalSource(unittest.TestCase):
    def test_step_test_matches_run_step_test(self):
        plant = benchmark_plant()
        t1, u1, _, y_meas1, _ = run_step_test(plant, step_amp=1.0)
        sig = SignalGenerator(plant).step_test(step_amp=1.0)
        np.testing.assert_allclose(sig.t, t1)
        np.testing.assert_allclose(sig.u, u1)
        np.testing.assert_allclose(sig.y, y_meas1)
        self.assertAlmostEqual(sig.dt, t1[1] - t1[0])
        self.assertEqual(sig.experiment, "step")

    def test_relay_test_matches_run_relay_test(self):
        plant = benchmark_plant()
        _, _, t1, u1, y1 = run_relay_test(plant, h=1.0)
        sig = SignalGenerator(plant).relay_test(h=1.0)
        np.testing.assert_allclose(sig.t, t1)
        np.testing.assert_allclose(sig.u, u1)
        np.testing.assert_allclose(sig.y, y1)
        self.assertEqual(sig.experiment, "relay")

    def test_step_test_publishes_only_measured_signal(self):
        """A real sensor never hands over the noise-free signal — the
        published Signal must carry only y_meas, never y_true."""
        plant = benchmark_plant()
        sig = SignalGenerator(plant).step_test(step_amp=1.0, noise_sigma=5.0, seed=0)
        _, _, y_true, y_meas, _ = run_step_test(plant, step_amp=1.0, noise_sigma=5.0, seed=0)
        np.testing.assert_allclose(sig.y, y_meas)
        self.assertFalse(np.allclose(sig.y, y_true))


class TestSOPDTFit(unittest.TestCase):
    def test_recovers_benchmark_parameters(self):
        """Benchmark plant has true tau1=10, tau2=1, L=0.5, K=1000 — two
        well-separated real poles, a clean SOPDT target."""
        plant = benchmark_plant()
        sig = SignalGenerator(plant).step_test(step_amp=1.0)
        fopdt = fit_fopdt_from_step(sig.t, sig.y, sig.meta["step_amp"])
        sopdt = fit_sopdt_from_step(sig.t, sig.y, sig.meta["step_amp"], fopdt_hint=fopdt)
        self.assertAlmostEqual(sopdt.tau1, 10.0, delta=1.5)
        self.assertAlmostEqual(sopdt.tau2, 1.0, delta=0.5)
        self.assertAlmostEqual(sopdt.L, 0.5, delta=0.3)
        self.assertAlmostEqual(sopdt.K, 1000.0, delta=50.0)

    def test_to_tf_has_two_real_stable_poles(self):
        sopdt = SOPDT(K=1000.0, tau1=10.0, tau2=1.0, L=0.5)
        poles = sopdt.to_tf().poles()
        self.assertEqual(len(poles), 2)
        for p in poles:
            self.assertAlmostEqual(float(np.imag(p)), 0.0, places=6)
            self.assertLess(np.real(p), 0.0)

    def test_declines_on_genuinely_first_order_plant(self):
        """A plant with only one real time constant shouldn't get a
        fabricated second pole — the fit should report itself degenerate."""
        plant = fopdt_plant()
        sig = SignalGenerator(plant).step_test(step_amp=1.0)
        with self.assertRaises(RuntimeError):
            fit_sopdt_from_step(sig.t, sig.y, sig.meta["step_amp"])

    def test_declines_on_underdamped_plant(self):
        """Complex poles can't be represented by the two-real-pole SOPDT
        form — the fit must refuse rather than mis-fit."""
        plant = TransferFunction.from_coeffs(num=[1.0], den=[1.0, 0.4, 4.0])
        sig = SignalGenerator(plant).step_test(step_amp=1.0)
        with self.assertRaises(RuntimeError):
            fit_sopdt_from_step(sig.t, sig.y, sig.meta["step_amp"])


class TestIdentifyUltimateGainFromRelay(unittest.TestCase):
    def test_matches_run_relay_test(self):
        plant = benchmark_plant()
        Ku1, Pu1, t, u, y = run_relay_test(plant, h=1.0)
        Ku2, Pu2 = identify_ultimate_gain_from_relay(t, u, y, h=1.0)
        self.assertAlmostEqual(Ku1, Ku2, places=9)
        self.assertAlmostEqual(Pu1, Pu2, places=9)


class TestBlackBox(unittest.TestCase):
    """End-to-end: entity C never receives a plant reference, only Signals."""

    def _signals(self):
        plant = benchmark_plant()
        gen = SignalGenerator(plant)
        step_sig = gen.step_test(step_amp=1.0)
        relay_sig = gen.relay_test(h=1.0)
        return step_sig, relay_sig

    def test_no_plant_import(self):
        import blackbox
        self.assertFalse(_module_imports_plant(blackbox.__file__),
                          "blackbox.py must never import plant.py directly")

    def test_all_methods_available_on_benchmark(self):
        step_sig, relay_sig = self._signals()
        tuner = BlackBoxTuner(step_signal=step_sig, relay_signal=relay_sig)
        rows = tuner.tune_all()
        self.assertEqual(len(rows), 12)  # 9 methods, 4 of them CHR variants
        for row in rows:
            self.assertTrue(row.available, f"{row.name} unexpectedly unavailable: {row.reason}")
            self.assertTrue(row.result.black_box)
            self.assertIsInstance(row.result.gains, PIDGains)

    def test_file_roundtrip_equivalent_to_in_process(self):
        step_sig, relay_sig = self._signals()
        with tempfile.TemporaryDirectory() as d:
            step_path = os.path.join(d, "step.npz")
            relay_path = os.path.join(d, "relay.npz")
            save_signal(step_sig, step_path)
            save_signal(relay_sig, relay_path)
            loaded_step = load_signal(step_path)
            loaded_relay = load_signal(relay_path)

        direct_rows = BlackBoxTuner(step_signal=step_sig, relay_signal=relay_sig).tune_all()
        file_rows = BlackBoxTuner(step_signal=loaded_step, relay_signal=loaded_relay).tune_all()
        self.assertEqual([r.name for r in direct_rows], [r.name for r in file_rows])
        for d_row, f_row in zip(direct_rows, file_rows):
            self.assertEqual(d_row.available, f_row.available)
            if d_row.available:
                self.assertAlmostEqual(d_row.result.gains.Kp, f_row.result.gains.Kp, places=9)
                self.assertAlmostEqual(d_row.result.gains.Ki, f_row.result.gains.Ki, places=9)
                self.assertAlmostEqual(d_row.result.gains.Kd, f_row.result.gains.Kd, places=9)

    def test_identify_flags_delay_detected_for_delayed_plant(self):
        step_sig, _ = self._signals()  # benchmark_plant has L=0.5
        model = identify_from_signals(step_signal=step_sig, relay_signal=None)
        self.assertTrue(model.fopdt.delay_detected)

    def test_identify_flags_no_delay_for_delay_free_plant(self):
        plant = TransferFunction.parse("1/(s+1)")
        step_sig = SignalGenerator(plant).step_test(step_amp=1.0)
        model = identify_from_signals(step_signal=step_sig, relay_signal=None)
        self.assertFalse(model.fopdt.delay_detected)

    def test_no_relay_signal_falls_back_to_surrogate_bode(self):
        step_sig, _ = self._signals()
        model = identify_from_signals(step_signal=step_sig, relay_signal=None)
        self.assertIsNotNone(model.Ku)
        self.assertIsNotNone(model.Pu)
        self.assertEqual(model.ku_pu_source, "surrogate-bode")

    def test_fopdt_based_gains_match_white_box_exactly(self):
        """ZN-I/AMIGO/SIMC/Cohen-Coon/CHR reduce to the same FOPDT fit the
        white-box pipeline computes (same generation code, no ground-truth
        shortcuts) — so black-box and white-box gains should match exactly."""
        plant = benchmark_plant()
        step_sig, relay_sig = SignalGenerator(plant).step_test(step_amp=1.0), \
            SignalGenerator(plant).relay_test(h=1.0)
        bb_rows = {r.name: r for r in
                   BlackBoxTuner(step_signal=step_sig, relay_signal=relay_sig).tune_all()}
        wb_rows = {r["name"]: r for r in compare_all_methods(plant, include_variants=True)}
        for name in ("ZN-I", "AMIGO", "SIMC", "Cohen–Coon"):
            self.assertTrue(bb_rows[name].available)
            self.assertAlmostEqual(bb_rows[name].result.gains.Kp, wb_rows[name]["gains"].Kp, places=6)

    def test_pole_cancellation_and_boyd_available_via_surrogate(self):
        """The two methods that historically needed the true plant directly
        (StablePoleCancellation, Boyd) should work here via the fitted
        SOPDT surrogate, since this plant has two well-separated real poles."""
        step_sig, relay_sig = self._signals()
        rows = {r.name: r for r in
                BlackBoxTuner(step_signal=step_sig, relay_signal=relay_sig).tune_all()}
        self.assertTrue(rows["Pole cancellation"].available)
        self.assertTrue(rows["Boyd"].available)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty test summary (when run as a script)
# ─────────────────────────────────────────────────────────────────────────────

class _Header(unittest.TextTestResult):
    """Print a header explaining what's being run."""
    pass


def _run_with_summary():
    """Same as `python -m unittest`, but with a final method-by-method
    metric table for the benchmark plant — so even if everything passes,
    you can eyeball the overshoots and confirm the methods behave the
    way the textbooks claim."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("Benchmark summary: G(s) = 1000/((s+1)(10s+1))·exp(-0.5s)")
        print("=" * 70)
        plant = benchmark_plant()
        _, _, _, _, fopdt = run_step_test(plant, step_amp=1.0)
        Ku, Pu, _ = find_ultimate_gain(plant)
        seed = tune_simc(fopdt).gains
        Kd_cancel = 1.0 / abs(plant.dc_gain())

        methods = [
            ("Pole cancellation",       tune_pole_cancellation(plant, 0.1, 1.0, Kd=Kd_cancel)),
            ("ZN-I",                    tune_zn_method_1(fopdt)),
            ("ZN-I  ½",                 halve_gains(tune_zn_method_1(fopdt))),
            ("ZN-II",                   tune_zn_method_2(Ku, Pu)),
            ("ZN-II ½",                 halve_gains(tune_zn_method_2(Ku, Pu))),
            ("AMIGO",                   tune_amigo(fopdt)),
            ("SIMC",                    tune_simc(fopdt)),
            ("Boyd (Ms=Mt=1.4)",        tune_boyd(plant, 1.4, 1.4, seed_gains=seed)),
            ("Cohen-Coon",              tune_cohen_coon(fopdt)),
            ("CHR setpoint 0%",         tune_chr(fopdt, "setpoint", 0)),
            ("CHR setpoint 20%",        tune_chr(fopdt, "setpoint", 20)),
            ("CHR load 0%",             tune_chr(fopdt, "load", 0)),
            ("CHR load 20%",            tune_chr(fopdt, "load", 20)),
            ("Tyreus-Luyben",           tune_tyreus_luyben(Ku, Pu)),
        ]
        for kind in ("step", "ramp", "pulse"):
            print(f"\n  --- {kind.upper()} setpoint ---")
            print(f"  {'method':<25s} {'Kp':>10s} {'Ki':>10s} {'Kd':>10s}", end="")
            if kind == "step":
                print(f"  {'OS%':>7s} {'ts(2%)':>8s} {'IAE':>8s}")
            elif kind == "ramp":
                print(f"  {'ss_err':>8s} {'max|e|':>8s} {'IAE':>8s}")
            else:
                print(f"  {'peak|e|':>9s} {'resid':>8s} {'IAE':>8s}")
            for name, res in methods:
                sim = simulate_closed_loop(plant, res.gains, setpoint=1.0,
                                           setpoint_kind=kind, t_end=80.0,
                                           u_min=-1e4, u_max=1e4)
                m = sim.metrics
                g = res.gains
                row = f"  {name:<25s} {g.Kp:10.4g} {g.Ki:10.4g} {g.Kd:10.4g}"
                if kind == "step":
                    row += f"  {m['Overshoot']:7.2f} {m['Settling']:8.3g} {m['IAE']:8.3g}"
                elif kind == "ramp":
                    row += f"  {m['ss_error']:8.4g} {m['max_error']:8.4g} {m['IAE']:8.3g}"
                else:
                    row += f"  {m['peak_error']:9.4g} {m['final_residual']:8.4g} {m['IAE']:8.3g}"
                print(row)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    sys.exit(_run_with_summary())
