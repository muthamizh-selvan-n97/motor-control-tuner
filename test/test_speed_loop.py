"""
test/test_speed_loop.py — pytest tests for modules/speed_loop.py

Verified analytical results:
  - pole_zero: Kp = J_total / (Kt_eff * tau_w_s)
  - root_locus: Kp = 2*zeta*omega_n*J_total/Kt_eff, Ki = omega_n^2*J_total/Kt_eff
  - frequency_domain: crossover placed within 20% of target BW_Hz
  - BW hierarchy: speed crossover < current BW / 10
  - anti-windup: Kb = Ki/Kp
  - all methods produce PM > 0, GM > 0
  - SPMSM and IPMSM both work
  - invalid method raises ValueError
  - plot methods save without crash
"""

import math
import pytest
import numpy as np

from utils.config import load_config
from modules.plant import PMSMPlant
from modules.param_id import ParameterIdentifier
from modules.current_loop import CurrentLoopTuner
from modules.speed_loop import SpeedLoopTuner, SpeedLoopResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spmsm_setup():
    motor_cfg, load_cfg = load_config(
        "config/motor_delta_ecma_c21010.yaml",
        "config/load_fan.yaml",
    )
    plant = PMSMPlant(motor_cfg, load_cfg)
    params = ParameterIdentifier(motor_cfg).simulate()
    i_loop = CurrentLoopTuner(plant, params).tune("pole_zero", axis="d", tau_cl_s=1e-3)
    return plant, params, i_loop


@pytest.fixture
def ipmsm_setup():
    motor_cfg, load_cfg = load_config(
        "config/motor_magnetic_blq40.yaml",
        "config/load_const_torque.yaml",
    )
    plant = PMSMPlant(motor_cfg, load_cfg)
    params = ParameterIdentifier(motor_cfg).simulate()
    i_loop = CurrentLoopTuner(plant, params).tune("pole_zero", axis="q", tau_cl_s=2e-3)
    return plant, params, i_loop


@pytest.fixture
def spmsm_tuner(spmsm_setup):
    plant, params, i_loop = spmsm_setup
    return SpeedLoopTuner(plant, params, i_loop)


@pytest.fixture
def ipmsm_tuner(ipmsm_setup):
    plant, params, i_loop = ipmsm_setup
    return SpeedLoopTuner(plant, params, i_loop)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_kt_eff_positive(self, spmsm_tuner):
        assert spmsm_tuner._Kt_eff > 0.0

    def test_j_total_greater_than_motor(self, spmsm_setup):
        plant, params, i_loop = spmsm_setup
        tuner = SpeedLoopTuner(plant, params, i_loop)
        J_motor = plant.motor_cfg["mechanical"]["J_kgm2"]
        assert tuner._J_total >= J_motor

    def test_fan_b_eff_greater_than_b_total(self, spmsm_tuner):
        """Fan load linearisation adds effective damping."""
        assert spmsm_tuner._B_eff >= spmsm_tuner._B_total

    def test_const_torque_b_eff_equals_b_total(self, ipmsm_tuner):
        """Constant torque load has no speed-dependent damping."""
        assert math.isclose(ipmsm_tuner._B_eff, ipmsm_tuner._B_total, rel_tol=1e-9)

    def test_tau_cl_positive(self, spmsm_tuner):
        assert spmsm_tuner._tau_cl > 0.0

    def test_tau_cl_from_bw(self, spmsm_setup):
        """tau_cl = 1 / (2*pi*BW_current) when BW > 0."""
        plant, params, i_loop = spmsm_setup
        tuner = SpeedLoopTuner(plant, params, i_loop)
        expected = 1.0 / (2.0 * math.pi * i_loop.BW_Hz)
        assert math.isclose(tuner._tau_cl, expected, rel_tol=1e-9)

    def test_kt_eff_formula(self, spmsm_setup):
        """Kt_eff = 1.5 * p * psi_f."""
        plant, params, i_loop = spmsm_setup
        tuner = SpeedLoopTuner(plant, params, i_loop)
        p = plant.p
        psi_f = params["psi_f"]
        expected = 1.5 * p * psi_f
        assert math.isclose(tuner._Kt_eff, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Invalid method
# ---------------------------------------------------------------------------

class TestInvalidMethod:
    def test_bad_method_raises(self, spmsm_tuner):
        with pytest.raises(ValueError, match="method must be one of"):
            spmsm_tuner.tune("bad_method")


# ---------------------------------------------------------------------------
# Pole-zero method
# ---------------------------------------------------------------------------

class TestPoleZero:
    def test_kp_formula(self, spmsm_tuner):
        """Kp = J_total / (Kt_eff * tau_w_s)."""
        tau_w = 10e-3
        res = spmsm_tuner.tune("pole_zero", tau_cl_s=tau_w)
        expected_kp = spmsm_tuner._J_total / (spmsm_tuner._Kt_eff * tau_w)
        assert math.isclose(res.Kp, expected_kp, rel_tol=1e-9)

    def test_ki_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert res.Ki > 0.0

    def test_method_label(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert res.method == "pole_zero"

    def test_axis_label(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert res.axis == "speed"

    def test_pm_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert res.PM_deg > 0.0

    def test_bw_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert res.BW_Hz > 0.0

    def test_bw_hierarchy(self, spmsm_setup):
        """Speed crossover must be below current BW / 10."""
        plant, params, i_loop = spmsm_setup
        tuner = SpeedLoopTuner(plant, params, i_loop)
        res = tuner.tune("pole_zero")
        assert res.crossover_Hz <= i_loop.BW_Hz / 10.0 + 0.5  # 0.5 Hz tolerance

    def test_bw_hierarchy_ipmsm(self, ipmsm_setup):
        plant, params, i_loop = ipmsm_setup
        tuner = SpeedLoopTuner(plant, params, i_loop)
        # Use a slow tau so speed crossover is well inside the hierarchy limit
        res = tuner.tune("pole_zero", tau_cl_s=50e-3)
        assert res.crossover_Hz <= i_loop.BW_Hz / 10.0 + 0.5

    def test_settling_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert res.settling_ms > 0.0

    def test_kb_equals_ki_over_kp(self, spmsm_tuner):
        """Anti-windup back-calculation gain Kb = Ki/Kp."""
        res = spmsm_tuner.tune("pole_zero")
        assert math.isclose(res.Kb, res.Ki / res.Kp, rel_tol=1e-9)

    def test_tau_w_larger_gives_smaller_kp(self, spmsm_tuner):
        """Larger tau_w → slower loop → smaller Kp."""
        res_fast = spmsm_tuner.tune("pole_zero", tau_cl_s=5e-3)
        res_slow = spmsm_tuner.tune("pole_zero", tau_cl_s=20e-3)
        assert res_slow.Kp < res_fast.Kp

    def test_returns_speed_loop_result(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert isinstance(res, SpeedLoopResult)


# ---------------------------------------------------------------------------
# Frequency-domain method
# ---------------------------------------------------------------------------

class TestFrequencyDomain:
    def test_method_label(self, spmsm_tuner):
        res = spmsm_tuner.tune("frequency_domain", BW_Hz=50)
        assert res.method == "frequency_domain"

    def test_crossover_near_target(self, spmsm_tuner):
        """Crossover frequency should be within 20% of target BW_Hz."""
        target = 50.0
        res = spmsm_tuner.tune("frequency_domain", BW_Hz=target)
        assert abs(res.crossover_Hz - target) / target < 0.20, \
            f"crossover={res.crossover_Hz:.1f} Hz vs target={target} Hz"

    def test_higher_bw_gives_higher_kp(self, spmsm_tuner):
        res_lo = spmsm_tuner.tune("frequency_domain", BW_Hz=20)
        res_hi = spmsm_tuner.tune("frequency_domain", BW_Hz=80)
        assert res_hi.Kp > res_lo.Kp

    def test_ki_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("frequency_domain", BW_Hz=50)
        assert res.Ki > 0.0

    def test_pm_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("frequency_domain", BW_Hz=30)
        assert res.PM_deg > 0.0


# ---------------------------------------------------------------------------
# Root-locus method
# ---------------------------------------------------------------------------

class TestRootLocus:
    def test_kp_formula(self, spmsm_tuner):
        """Kp = 2*zeta*omega_n*J_total/Kt_eff."""
        zeta = 0.707
        tau_w = 10e-3
        res = spmsm_tuner.tune("root_locus", target_zeta=zeta, tau_cl_s=tau_w)
        omega_n = 1.0 / tau_w
        expected_kp = 2.0 * zeta * omega_n * spmsm_tuner._J_total / spmsm_tuner._Kt_eff
        assert math.isclose(res.Kp, expected_kp, rel_tol=1e-9)

    def test_ki_formula(self, spmsm_tuner):
        """Ki = omega_n^2 * J_total / Kt_eff."""
        tau_w = 10e-3
        res = spmsm_tuner.tune("root_locus", tau_cl_s=tau_w)
        omega_n = 1.0 / tau_w
        expected_ki = omega_n ** 2 * spmsm_tuner._J_total / spmsm_tuner._Kt_eff
        assert math.isclose(res.Ki, expected_ki, rel_tol=1e-9)

    def test_method_label(self, spmsm_tuner):
        res = spmsm_tuner.tune("root_locus")
        assert res.method == "root_locus"

    def test_pm_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("root_locus")
        assert res.PM_deg > 0.0

    def test_higher_zeta_different_kp(self, spmsm_tuner):
        res_lo = spmsm_tuner.tune("root_locus", target_zeta=0.5, tau_cl_s=10e-3)
        res_hi = spmsm_tuner.tune("root_locus", target_zeta=1.0, tau_cl_s=10e-3)
        assert res_hi.Kp != res_lo.Kp


# ---------------------------------------------------------------------------
# Ziegler-Nichols method
# ---------------------------------------------------------------------------

class TestZieglerNichols:
    def test_method_label(self, spmsm_tuner):
        res = spmsm_tuner.tune("ziegler_nichols")
        assert res.method == "ziegler_nichols"

    def test_kp_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("ziegler_nichols")
        assert res.Kp > 0.0

    def test_ki_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("ziegler_nichols")
        assert res.Ki > 0.0

    def test_pm_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("ziegler_nichols")
        assert res.PM_deg > 0.0

    def test_ipmsm_zn(self, ipmsm_tuner):
        res = ipmsm_tuner.tune("ziegler_nichols")
        assert res.Kp > 0.0
        assert res.Ki > 0.0


# ---------------------------------------------------------------------------
# LoopResult structure
# ---------------------------------------------------------------------------

class TestLoopResultStructure:
    def test_has_all_fields(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        for attr in ("Kp", "Ki", "method", "axis", "BW_Hz", "PM_deg",
                     "GM_dB", "crossover_Hz", "settling_ms", "overshoot_pct",
                     "warnings", "Kb", "Kt_eff", "J_total", "B_eff", "tau_cl"):
            assert hasattr(res, attr), f"Missing attribute: {attr}"

    def test_warnings_is_list(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert isinstance(res.warnings, list)

    def test_gm_non_negative(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert res.GM_dB >= 0.0 or res.GM_dB > 900

    def test_bw_hz_non_negative(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert res.BW_Hz >= 0.0

    def test_kt_eff_stored(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert math.isclose(res.Kt_eff, spmsm_tuner._Kt_eff, rel_tol=1e-9)

    def test_j_total_stored(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert math.isclose(res.J_total, spmsm_tuner._J_total, rel_tol=1e-9)

    def test_tau_cl_stored(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert math.isclose(res.tau_cl, spmsm_tuner._tau_cl, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# BW hierarchy warnings
# ---------------------------------------------------------------------------

class TestBWHierarchy:
    def test_no_warning_for_conservative_tau(self, spmsm_tuner):
        """pole_zero at slow tau (50 ms) should not warn about BW hierarchy."""
        res = spmsm_tuner.tune("pole_zero", tau_cl_s=50e-3)
        bw_warns = [w for w in res.warnings if "BW" in w or "Speed" in w]
        assert len(bw_warns) == 0

    def test_aggressive_bw_triggers_warning(self, spmsm_setup):
        """Very aggressive speed loop BW should trigger hierarchy warning."""
        plant, params, i_loop = spmsm_setup
        # Make current loop with very low BW
        low_bw_loop = CurrentLoopTuner(plant, params).tune("pole_zero", axis="d",
                                                            tau_cl_s=50e-3)
        tuner = SpeedLoopTuner(plant, params, low_bw_loop)
        # Tune speed loop with very aggressive BW (above current/10)
        res = tuner.tune("frequency_domain", BW_Hz=low_bw_loop.BW_Hz * 0.5)
        bw_warns = [w for w in res.warnings if "BW" in w or "Speed" in w]
        assert len(bw_warns) > 0


# ---------------------------------------------------------------------------
# Anti-windup
# ---------------------------------------------------------------------------

class TestAntiWindup:
    def test_kb_equals_ki_over_kp_all_methods(self, spmsm_tuner):
        for method in ("pole_zero", "frequency_domain", "root_locus",
                       "ziegler_nichols"):
            res = spmsm_tuner.tune(method)
            assert math.isclose(res.Kb, res.Ki / res.Kp, rel_tol=1e-9), \
                f"Method {method}: Kb={res.Kb:.6f} != Ki/Kp={res.Ki/res.Kp:.6f}"

    def test_kb_positive(self, spmsm_tuner):
        res = spmsm_tuner.tune("pole_zero")
        assert res.Kb > 0.0


# ---------------------------------------------------------------------------
# Summary output (smoke test)
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_runs(self, spmsm_tuner, capsys):
        res = spmsm_tuner.tune("pole_zero")
        res.summary()
        captured = capsys.readouterr()
        assert "Kp" in captured.out
        assert "speed" in captured.out

    def test_summary_all_methods(self, spmsm_tuner, capsys):
        for method in ("pole_zero", "frequency_domain", "root_locus",
                       "ziegler_nichols"):
            res = spmsm_tuner.tune(method)
            res.summary()
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ---------------------------------------------------------------------------
# Plots (smoke tests)
# ---------------------------------------------------------------------------

class TestPlots:
    def test_bode_saves(self, spmsm_tuner, tmp_path):
        res = spmsm_tuner.tune("pole_zero")
        spmsm_tuner.plot_bode(res, save_path=str(tmp_path / "bode_speed.png"))

    def test_step_saves(self, spmsm_tuner, tmp_path):
        res = spmsm_tuner.tune("pole_zero")
        spmsm_tuner.plot_step(res, save_path=str(tmp_path / "step_speed.png"))

    def test_antiwindup_saves(self, spmsm_tuner, tmp_path):
        res = spmsm_tuner.tune("pole_zero")
        spmsm_tuner.plot_antiwindup(res,
                                    save_path=str(tmp_path / "aw_speed.png"))

    def test_ipmsm_bode_saves(self, ipmsm_tuner, tmp_path):
        res = ipmsm_tuner.tune("pole_zero")
        ipmsm_tuner.plot_bode(res, save_path=str(tmp_path / "ipmsm_bode.png"))

    @pytest.mark.parametrize("method", [
        "pole_zero", "frequency_domain", "root_locus", "ziegler_nichols"
    ])
    def test_all_methods_bode_and_step(self, spmsm_tuner, method, tmp_path):
        res = spmsm_tuner.tune(method)
        spmsm_tuner.plot_bode(res, save_path=str(tmp_path / f"bode_{method}.png"))
        spmsm_tuner.plot_step(res, save_path=str(tmp_path / f"step_{method}.png"))
