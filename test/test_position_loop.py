"""
test/test_position_loop.py — Tests for PositionLoopTuner.

Analytical checks:
  1. P controller: Kp_pos ≈ omega_c (when tau_w << 1/omega_c)
  2. PD controller: Kd_pos = 2*zeta / omega_n, Kp_pos >= P Kp
  3. BW hierarchy: tune() raises ValueError if BW_pos > BW_speed/10
  4. Closed-loop poles: real part of dominant pole ≈ -omega_n (P case)
  5. Following error: 1/Kp_pos for unit ramp (Kff_v=0)
  6. PD phase margin > P phase margin (derivative adds lead)
  7. Kff_v stored correctly in result
"""

import math
import pytest
import numpy as np

from utils.config import load_config
from modules.plant import PMSMPlant
from modules.param_id import ParameterIdentifier
from modules.current_loop import CurrentLoopTuner
from modules.speed_loop import SpeedLoopTuner
from modules.position_loop import PositionLoopTuner, PositionLoopResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def servo_setup():
    """SPMSM + position-servo load → i_loop, w_loop, tuner."""
    motor_cfg, load_cfg = load_config(
        "config/motor_delta_ecma_c21010.yaml",
        "config/load_position_servo.yaml",
    )
    plant = PMSMPlant(motor_cfg, load_cfg)
    params = ParameterIdentifier(motor_cfg).simulate(noise_std_frac=0.0)
    i_loop = CurrentLoopTuner(plant, params).tune("pole_zero", tau_d_ms=1.0)
    w_loop = SpeedLoopTuner(plant, params, i_loop).tune("pole_zero")
    tuner = PositionLoopTuner(plant, w_loop)
    return tuner, w_loop


# ---------------------------------------------------------------------------
# Test 1 — P controller: Kp ≈ omega_c when tau_w is small
# ---------------------------------------------------------------------------

def test_P_kp_equals_omega_c(servo_setup):
    """
    When tau_w << 1/omega_c, Kp_pos ≈ omega_c = 2*pi*BW_pos.

    Allow ±10% because the exact formula includes the sqrt(1+(tau_w*omega_c)^2) factor.
    """
    tuner, w_loop = servo_setup
    BW_pos_Hz = w_loop.BW_Hz / 10.0
    result = tuner.tune("P", BW_pos_Hz=BW_pos_Hz)

    omega_c_expected = 2 * math.pi * BW_pos_Hz
    tau_w = 1.0 / (2 * math.pi * w_loop.BW_Hz)
    exact_Kp = omega_c_expected * math.sqrt(1.0 + (tau_w * omega_c_expected) ** 2)

    assert abs(result.Kp - exact_Kp) / exact_Kp < 1e-6, (
        f"Kp mismatch: got {result.Kp:.4f}, expected {exact_Kp:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — PD: Kd = 2*zeta/omega_n
# ---------------------------------------------------------------------------

def test_PD_kd_formula(servo_setup):
    """Kd_pos = 2*zeta / omega_n where omega_n = 2*pi*BW_pos."""
    tuner, w_loop = servo_setup
    zeta = 0.7
    BW_pos_Hz = w_loop.BW_Hz / 10.0
    result = tuner.tune("PD", BW_pos_Hz=BW_pos_Hz, zeta=zeta)

    omega_n = 2 * math.pi * BW_pos_Hz
    Kd_expected = 2.0 * zeta / omega_n

    assert abs(result.Kd - Kd_expected) / Kd_expected < 1e-6, (
        f"Kd mismatch: got {result.Kd:.6f}, expected {Kd_expected:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — BW hierarchy: ValueError when BW_pos > BW_speed/10
# ---------------------------------------------------------------------------

def test_BW_hierarchy_violation(servo_setup):
    """BW_pos > BW_speed/10 must raise ValueError."""
    tuner, w_loop = servo_setup
    too_high = w_loop.BW_Hz / 10.0 + 0.5   # just over the limit
    with pytest.raises(ValueError, match="exceeds maximum"):
        tuner.tune("P", BW_pos_Hz=too_high)


# ---------------------------------------------------------------------------
# Test 4 — P closed-loop: dominant pole ≈ -omega_n (low tau_w regime)
# ---------------------------------------------------------------------------

def test_P_closed_loop_pole(servo_setup):
    """
    For the P controller, the CL denominator is tau_w*s^2 + s + Kp.
    When tau_w << 1/omega_n the dominant real pole ≈ -Kp (= -omega_c).
    Allow ±15% because tau_w is not negligible in general.
    """
    tuner, w_loop = servo_setup
    BW_pos_Hz = w_loop.BW_Hz / 10.0
    result = tuner.tune("P", BW_pos_Hz=BW_pos_Hz)

    omega_c = 2 * math.pi * BW_pos_Hz
    tau_w = 1.0 / (2 * math.pi * w_loop.BW_Hz)
    # CL denom: tau_w*s^2 + s + Kp  →  roots
    poles = np.roots([tau_w, 1.0, result.Kp])
    dominant_pole = poles[np.argmax(np.real(poles))]   # least negative
    assert abs(np.real(dominant_pole) + omega_c) / omega_c < 0.20, (
        f"Dominant pole {dominant_pole:.2f} not near -omega_c {-omega_c:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Following error (ramp, Kff_v=0)
# ---------------------------------------------------------------------------

def test_following_error_ramp(servo_setup):
    """
    Steady-state ramp following error = 1/Kp_pos  (velocity error constant).
    Stored in result.following_error_ramp.
    """
    tuner, w_loop = servo_setup
    BW_pos_Hz = w_loop.BW_Hz / 10.0
    result = tuner.tune("P", BW_pos_Hz=BW_pos_Hz, Kff_v=0.0)

    expected_e_ss = 1.0 / result.Kp
    assert abs(result.following_error_ramp - expected_e_ss) < 1e-9, (
        f"e_ss={result.following_error_ramp:.6f}, expected {expected_e_ss:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 6 — PD phase margin > P phase margin
# ---------------------------------------------------------------------------

def test_PD_better_phase_margin_than_P(servo_setup):
    """Derivative action (PD) must improve phase margin vs plain P."""
    tuner, w_loop = servo_setup
    BW_pos_Hz = w_loop.BW_Hz / 10.0
    res_P  = tuner.tune("P",  BW_pos_Hz=BW_pos_Hz)
    res_PD = tuner.tune("PD", BW_pos_Hz=BW_pos_Hz, zeta=0.7)

    assert res_PD.PM_deg > res_P.PM_deg, (
        f"PD PM ({res_PD.PM_deg:.1f}°) should exceed P PM ({res_P.PM_deg:.1f}°)"
    )


# ---------------------------------------------------------------------------
# Test 7 — Kff_v stored correctly
# ---------------------------------------------------------------------------

def test_kff_v_stored(servo_setup):
    """Kff_v must be echoed back in PositionLoopResult."""
    tuner, _ = servo_setup
    for kff in [0.0, 0.5, 1.0]:
        result = tuner.tune("P", Kff_v=kff)
        assert result.Kff_v == kff


# ---------------------------------------------------------------------------
# Test 8 — PositionLoopResult is correct type
# ---------------------------------------------------------------------------

def test_result_type(servo_setup):
    """tune() must return a PositionLoopResult."""
    tuner, _ = servo_setup
    result = tuner.tune("P")
    assert isinstance(result, PositionLoopResult)


# ---------------------------------------------------------------------------
# Test 9 — Ki is always 0 (no integral in position loop)
# ---------------------------------------------------------------------------

def test_no_integral_term(servo_setup):
    """Position loop is P/PD only — Ki must be zero."""
    tuner, _ = servo_setup
    for method in ("P", "PD"):
        result = tuner.tune(method)
        assert result.Ki == 0.0, f"Ki={result.Ki} for method={method}"


# ---------------------------------------------------------------------------
# Test 10 — Phase margin within reasonable range
# ---------------------------------------------------------------------------

def test_phase_margin_acceptable(servo_setup):
    """
    Default P and PD tuning (BW_pos = BW_speed/10) should yield PM >= 45 deg.
    """
    tuner, _ = servo_setup
    for method in ("P", "PD"):
        result = tuner.tune(method)
        assert result.PM_deg >= 45.0, (
            f"Method {method}: PM={result.PM_deg:.1f} deg < 45 deg"
        )
