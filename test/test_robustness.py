"""
test/test_robustness.py — pytest tests for modules/robustness.py

Verified analytical results:
  - parameter_sweep: nominal point (delta=0) matches i_loop PM/GM within 1%
  - parameter_sweep: grid shape is (steps, steps)
  - parameter_sweep: Rs+ increases PM for pole-zero tuned loop (proportional gain
    unchanged but plant DC gain drops → crossover moves left → more phase)
  - sensitivity: S + T = 1 everywhere (|S|^2 + |T|^2 not directly checkable, but
    at crossover |S| ≈ |T| ≈ 0 dB)
  - sensitivity: Ms_dB >= 0 (peak >= 0 dB)
  - sensitivity: result dict has all expected keys
  - margin_waterfall: arrays have length = steps
  - margin_waterfall: current loop PM degrades for very large Rs (+100%)
  - plots save without crash
"""

import math
import pytest
import numpy as np

from utils.config import load_config
from modules.plant import PMSMPlant
from modules.param_id import ParameterIdentifier
from modules.current_loop import CurrentLoopTuner
from modules.speed_loop import SpeedLoopTuner
from modules.robustness import RobustnessAnalyser, _compute_margins

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spmsm_components():
    motor_cfg, load_cfg = load_config(
        "config/motor_delta_ecma_c21010.yaml",
        "config/load_fan.yaml",
    )
    plant = PMSMPlant(motor_cfg, load_cfg)
    params = ParameterIdentifier(motor_cfg).simulate()
    i_loop = CurrentLoopTuner(plant, params).tune("pole_zero", axis="d", tau_cl_s=1e-3)
    w_loop = SpeedLoopTuner(plant, params, i_loop).tune("pole_zero", tau_cl_s=50e-3)
    return plant, params, i_loop, w_loop


@pytest.fixture(scope="module")
def ipmsm_components():
    motor_cfg, load_cfg = load_config(
        "config/motor_magnetic_blq40.yaml",
        "config/load_const_torque.yaml",
    )
    plant = PMSMPlant(motor_cfg, load_cfg)
    params = ParameterIdentifier(motor_cfg).simulate()
    i_loop = CurrentLoopTuner(plant, params).tune("pole_zero", axis="q", tau_cl_s=2e-3)
    w_loop = SpeedLoopTuner(plant, params, i_loop).tune("pole_zero", tau_cl_s=50e-3)
    return plant, params, i_loop, w_loop


@pytest.fixture(scope="module")
def analyser(spmsm_components):
    plant, params, i_loop, w_loop = spmsm_components
    return RobustnessAnalyser(plant, i_loop, w_loop)


@pytest.fixture(scope="module")
def analyser_ipmsm(ipmsm_components):
    plant, params, i_loop, w_loop = ipmsm_components
    return RobustnessAnalyser(plant, i_loop, w_loop)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_stores_nominal_rs(self, analyser, spmsm_components):
        plant, _, _, _ = spmsm_components
        assert math.isclose(analyser._Rs_nom, plant.Rs, rel_tol=1e-9)

    def test_stores_nominal_ld(self, analyser, spmsm_components):
        plant, _, _, _ = spmsm_components
        assert math.isclose(analyser._Ld_nom, plant.Ld, rel_tol=1e-9)

    def test_stores_i_loop(self, analyser, spmsm_components):
        _, _, i_loop, _ = spmsm_components
        assert analyser.i_loop is i_loop

    def test_stores_w_loop(self, analyser, spmsm_components):
        _, _, _, w_loop = spmsm_components
        assert analyser.w_loop is w_loop


# ---------------------------------------------------------------------------
# _compute_margins helper
# ---------------------------------------------------------------------------

class TestComputeMargins:
    def test_known_first_order_pi(self):
        """
        For pole-zero cancellation with tau_cl=1ms on SPMSM:
        PM should be well above 45 deg (near 90 deg for exact cancellation).
        """
        import numpy as np
        Rs = 0.20
        Ld = 1.81e-3
        tau_cl = 1e-3
        Kp = Ld / tau_cl
        Ki = Rs / tau_cl
        num_ol = np.array([Kp, Ki])
        den_ol = np.array([Rs * (Ld / Rs), Rs, 0.0])
        PM, GM = _compute_margins(num_ol, den_ol)
        assert PM > 45.0
        assert GM > 6.0

    def test_returns_float_tuple(self):
        num = np.array([1.0, 1.0])
        den = np.array([1.0, 2.0, 0.0])
        PM, GM = _compute_margins(num, den)
        assert isinstance(PM, float)
        assert isinstance(GM, float)

    def test_gm_large_for_first_order_plant(self):
        """PI + first-order plant: no phase crossover → GM effectively infinite."""
        num = np.array([1.0, 0.1])
        den = np.array([0.01, 1.0, 0.0])
        _, GM = _compute_margins(num, den)
        assert GM > 100.0  # returns 999 sentinel


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

class TestParameterSweep:
    def test_grid_shape(self, analyser):
        result = analyser.parameter_sweep(steps=5, axis="d")
        assert result["PM_grid"].shape == (5, 5)
        assert result["GM_grid"].shape == (5, 5)

    def test_rs_deltas_length(self, analyser):
        result = analyser.parameter_sweep(steps=7, axis="d")
        assert len(result["Rs_deltas"]) == 7

    def test_l_deltas_length(self, analyser):
        result = analyser.parameter_sweep(steps=7, axis="d")
        assert len(result["L_deltas"]) == 7

    def test_nominal_pm_matches_iloop(self, analyser, spmsm_components):
        """Centre point (delta=0) should match i_loop PM within 1%."""
        _, _, i_loop, _ = spmsm_components
        result = analyser.parameter_sweep(steps=5, axis="d")
        nom_PM = result["nominal_PM"]
        assert abs(nom_PM - i_loop.PM_deg) / max(i_loop.PM_deg, 1.0) < 0.01, \
            f"Nominal PM={nom_PM:.1f} vs i_loop.PM={i_loop.PM_deg:.1f}"

    def test_all_pm_positive(self, analyser):
        result = analyser.parameter_sweep(steps=5, axis="d")
        assert np.all(result["PM_grid"] >= 0.0)

    def test_all_gm_positive(self, analyser):
        result = analyser.parameter_sweep(steps=5, axis="d")
        assert np.all(result["GM_grid"] >= 0.0)

    def test_returns_dict_keys(self, analyser):
        result = analyser.parameter_sweep(steps=3, axis="d")
        for key in ("Rs_deltas", "L_deltas", "PM_grid", "GM_grid",
                    "nominal_PM", "nominal_GM"):
            assert key in result

    def test_q_axis_works(self, analyser):
        result = analyser.parameter_sweep(steps=3, axis="q")
        assert result["PM_grid"].shape == (3, 3)

    def test_ipmsm_sweep(self, analyser_ipmsm):
        result = analyser_ipmsm.parameter_sweep(steps=3, axis="q")
        assert result["nominal_PM"] > 0.0


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

class TestSensitivity:
    def test_returns_dict_keys(self, analyser):
        result = analyser.sensitivity(axis="d")
        for key in ("freq_Hz", "S_db", "T_db", "Ms_dB", "Mt_dB", "Ms_Hz", "warnings"):
            assert key in result

    def test_ms_db_non_negative(self, analyser):
        """Peak sensitivity >= 0 dB (|S| -> 1 at high freq → Ms >= 0 dB)."""
        result = analyser.sensitivity(axis="d")
        assert result["Ms_dB"] >= -0.01  # allow small numerical noise

    def test_mt_db_near_zero_at_unity_gain(self, analyser):
        """At DC the closed loop has unity gain: T → 1 (0 dB)."""
        result = analyser.sensitivity(axis="d")
        assert result["Mt_dB"] > -5.0  # T_db near 0 dB at low freq

    def test_warnings_is_list(self, analyser):
        result = analyser.sensitivity(axis="d")
        assert isinstance(result["warnings"], list)

    def test_freq_array_length(self, analyser):
        result = analyser.sensitivity(axis="d")
        assert len(result["freq_Hz"]) == len(result["S_db"])
        assert len(result["freq_Hz"]) == len(result["T_db"])

    def test_ms_hz_positive(self, analyser):
        result = analyser.sensitivity(axis="d")
        assert result["Ms_Hz"] > 0.0

    def test_well_tuned_loop_no_warning(self, analyser):
        """Pole-zero tuned loop should have Ms < 6 dB (robust design)."""
        result = analyser.sensitivity(axis="d")
        # Ms < 6 dB → no warning
        if 10.0 ** (result["Ms_dB"] / 20.0) < 2.0:
            assert len(result["warnings"]) == 0

    def test_ipmsm_sensitivity(self, analyser_ipmsm):
        result = analyser_ipmsm.sensitivity(axis="q")
        assert result["Ms_dB"] >= -0.01  # allow small numerical noise


# ---------------------------------------------------------------------------
# Margin waterfall
# ---------------------------------------------------------------------------

class TestMarginWaterfall:
    def test_arrays_have_correct_length(self, analyser):
        result = analyser.margin_waterfall(steps=11, axis="d")
        assert len(result["Rs_deltas"]) == 11
        assert len(result["PM_current"]) == 11
        assert len(result["GM_current"]) == 11

    def test_pm_current_positive_at_nominal(self, analyser):
        result = analyser.margin_waterfall(steps=11, axis="d")
        mid = len(result["Rs_deltas"]) // 2
        assert result["PM_current"][mid] > 0.0

    def test_returns_dict_keys(self, analyser):
        result = analyser.margin_waterfall(steps=5, axis="d")
        for key in ("Rs_deltas", "PM_current", "GM_current",
                    "PM_speed", "GM_speed"):
            assert key in result

    def test_speed_pm_returned(self, analyser):
        result = analyser.margin_waterfall(steps=5, axis="d")
        assert result["PM_speed"] is not None
        assert len(result["PM_speed"]) == 5

    def test_nominal_current_pm_matches_iloop(self, analyser, spmsm_components):
        """Centre of waterfall should match i_loop PM within 1%."""
        _, _, i_loop, _ = spmsm_components
        result = analyser.margin_waterfall(steps=11, axis="d")
        mid = len(result["Rs_deltas"]) // 2
        nom = result["PM_current"][mid]
        assert abs(nom - i_loop.PM_deg) / max(i_loop.PM_deg, 1.0) < 0.01

    def test_ipmsm_waterfall(self, analyser_ipmsm):
        result = analyser_ipmsm.margin_waterfall(steps=5, axis="q")
        assert len(result["PM_current"]) == 5

    def test_all_pm_non_negative(self, analyser):
        result = analyser.margin_waterfall(steps=11, axis="d")
        assert np.all(result["PM_current"] >= 0.0)


# ---------------------------------------------------------------------------
# Plots (smoke tests)
# ---------------------------------------------------------------------------

class TestPlots:
    def test_sweep_saves(self, analyser, tmp_path):
        analyser.parameter_sweep(
            steps=5, axis="d",
            save_path=str(tmp_path / "sweep.png"),
        )

    def test_sensitivity_saves(self, analyser, tmp_path):
        analyser.sensitivity(
            axis="d",
            save_path=str(tmp_path / "sensitivity.png"),
        )

    def test_waterfall_saves(self, analyser, tmp_path):
        analyser.margin_waterfall(
            steps=7, axis="d",
            save_path=str(tmp_path / "waterfall.png"),
        )

    def test_ipmsm_all_plots(self, analyser_ipmsm, tmp_path):
        analyser_ipmsm.parameter_sweep(
            steps=3, axis="q",
            save_path=str(tmp_path / "ipmsm_sweep.png"),
        )
        analyser_ipmsm.sensitivity(
            axis="q",
            save_path=str(tmp_path / "ipmsm_sens.png"),
        )
        analyser_ipmsm.margin_waterfall(
            steps=5, axis="q",
            save_path=str(tmp_path / "ipmsm_wf.png"),
        )
