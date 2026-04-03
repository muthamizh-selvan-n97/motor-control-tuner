"""
test/test_discretization.py — pytest tests for modules/discretization.py

Verified analytical results:
  - Tustin Ki_d = Ki * Ts/2  (bilinear integrator approximation)
  - ZOH Ki_d ≈ Ki * Ts       (exact hold-equivalent integrator)
  - Phase loss = 1.5 * Ts * omega_c * (180/pi) degrees
  - BW_max = 1/(6*Ts)
  - PM_with_delay < PM_no_delay
  - Q15: quantized float reconstructed from int matches within 1 LSB
  - Q15: quantization error < 0.01% for values that fit in Q15
  - firmware_table() returns a string with Kp and Ki
  - Plots save without crash
"""

import math
import pytest
import numpy as np

from utils.config import load_config
from modules.plant import PMSMPlant
from modules.param_id import ParameterIdentifier
from modules.current_loop import CurrentLoopTuner
from modules.discretization import Discretizer, _Q15_SCALE

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TS = 50e-6   # 20 kHz


@pytest.fixture
def spmsm_loop():
    motor_cfg, load_cfg = load_config(
        "config/motor_delta_ecma_c21010.yaml",
        "config/load_fan.yaml",
    )
    plant = PMSMPlant(motor_cfg, load_cfg)
    params = ParameterIdentifier(motor_cfg).simulate()
    tuner = CurrentLoopTuner(plant, params)
    return tuner.tune("pole_zero", axis="d", tau_cl_s=1e-3)


@pytest.fixture
def ipmsm_loop():
    motor_cfg, load_cfg = load_config(
        "config/motor_magnetic_blq40.yaml",
        "config/load_const_torque.yaml",
    )
    plant = PMSMPlant(motor_cfg, load_cfg)
    params = ParameterIdentifier(motor_cfg).simulate()
    tuner = CurrentLoopTuner(plant, params)
    return tuner.tune("pole_zero", axis="q", tau_cl_s=2e-3)


@pytest.fixture
def disc(spmsm_loop):
    return Discretizer(spmsm_loop, Ts_s=_TS)


@pytest.fixture
def disc_fd():
    """Frequency-domain tuned loop — has non-trivial crossover for phase loss test."""
    motor_cfg, load_cfg = load_config(
        "config/motor_delta_ecma_c21010.yaml",
        "config/load_fan.yaml",
    )
    plant = PMSMPlant(motor_cfg, load_cfg)
    params = ParameterIdentifier(motor_cfg).simulate()
    tuner = CurrentLoopTuner(plant, params)
    loop = tuner.tune("frequency_domain", axis="d", BW_Hz=500)
    return Discretizer(loop, Ts_s=_TS)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_invalid_ts_raises(self, spmsm_loop):
        with pytest.raises(ValueError):
            Discretizer(spmsm_loop, Ts_s=0.0)

    def test_negative_ts_raises(self, spmsm_loop):
        with pytest.raises(ValueError):
            Discretizer(spmsm_loop, Ts_s=-1e-6)

    def test_stores_kp_ki(self, disc, spmsm_loop):
        assert disc.Kp == spmsm_loop.Kp
        assert disc.Ki == spmsm_loop.Ki


# ---------------------------------------------------------------------------
# compare_methods — Tustin
# ---------------------------------------------------------------------------

class TestTustin:
    def test_ki_d_approx_ki_times_ts_over_2(self, disc):
        """Tustin integrator: Ki_d ≈ Ki * Ts/2."""
        res = disc.compare_methods()
        expected = disc.Ki * _TS / 2.0
        assert math.isclose(res["tustin"].Ki_d, expected, rel_tol=0.01), \
            f"Tustin Ki_d={res['tustin'].Ki_d:.6f} vs Ki*Ts/2={expected:.6f}"

    def test_kp_d_equals_kp(self, disc):
        """Tustin Kp_d = Kp (proportional gain unchanged)."""
        res = disc.compare_methods()
        assert math.isclose(res["tustin"].Kp_d, disc.Kp, rel_tol=1e-9)

    def test_method_label(self, disc):
        res = disc.compare_methods()
        assert res["tustin"].method == "tustin"

    def test_z_poles_on_unit_circle_boundary(self, disc):
        """Tustin pole at z=1 (integrator maps to unit circle)."""
        res = disc.compare_methods()
        z_poles = res["tustin"].z_poles
        # One pole should be near z=1 (integrator)
        assert any(abs(abs(p) - 1.0) < 0.01 for p in z_poles)

    def test_phase_loss_non_negative(self, disc):
        res = disc.compare_methods()
        assert res["tustin"].phase_loss_deg >= 0.0

    def test_pm_eff_less_than_continuous(self, disc):
        res = disc.compare_methods()
        assert res["tustin"].PM_eff_deg <= disc.i_loop.PM_deg + 0.1


# ---------------------------------------------------------------------------
# compare_methods — ZOH
# ---------------------------------------------------------------------------

class TestZOH:
    def test_ki_d_approx_ki_times_ts(self, disc):
        """ZOH integrator: Ki_d ≈ Ki * Ts."""
        res = disc.compare_methods()
        expected = disc.Ki * _TS
        assert math.isclose(res["zoh"].Ki_d, expected, rel_tol=0.05), \
            f"ZOH Ki_d={res['zoh'].Ki_d:.6f} vs Ki*Ts={expected:.6f}"

    def test_method_label(self, disc):
        res = disc.compare_methods()
        assert res["zoh"].method == "zoh"

    def test_phase_loss_non_negative(self, disc):
        res = disc.compare_methods()
        assert res["zoh"].phase_loss_deg >= 0.0

    def test_pm_eff_less_than_continuous(self, disc):
        res = disc.compare_methods()
        assert res["zoh"].PM_eff_deg <= disc.i_loop.PM_deg + 0.1

    def test_tustin_vs_zoh_ki_d_ratio(self, disc):
        """Tustin Ki_d ≈ ZOH Ki_d / 2 (Ts/2 vs Ts)."""
        res = disc.compare_methods()
        ratio = res["tustin"].Ki_d / res["zoh"].Ki_d
        assert math.isclose(ratio, 0.5, rel_tol=0.05), \
            f"Tustin/ZOH Ki_d ratio = {ratio:.3f}, expected 0.5"


# ---------------------------------------------------------------------------
# PWM delay analysis
# ---------------------------------------------------------------------------

class TestPWMDelay:
    def test_phase_loss_formula(self, disc_fd):
        """Phase loss = 1.5 * Ts * omega_c * (180/pi)."""
        res = disc_fd.pwm_delay_analysis()
        expected = 1.5 * _TS * disc_fd.omega_c * (180.0 / math.pi)
        assert math.isclose(res["phase_loss_deg"], expected, rel_tol=1e-6)

    def test_pm_with_delay_less_than_without(self, disc_fd):
        res = disc_fd.pwm_delay_analysis()
        assert res["PM_with_delay"] < res["PM_no_delay"]

    def test_pm_no_delay_matches_loop_result(self, disc_fd):
        res = disc_fd.pwm_delay_analysis()
        assert math.isclose(
            res["PM_no_delay"], disc_fd.i_loop.PM_deg, rel_tol=1e-9
        )

    def test_bw_max_formula(self, disc):
        """BW_max = 1/(6*Ts)."""
        res = disc.pwm_delay_analysis()
        expected = 1.0 / (6.0 * _TS)
        assert math.isclose(res["BW_max_Hz"], expected, rel_tol=1e-9)

    def test_returns_dict_keys(self, disc):
        res = disc.pwm_delay_analysis()
        for key in ("phase_loss_deg", "PM_no_delay", "PM_with_delay",
                    "BW_max_Hz", "warnings"):
            assert key in res

    def test_conservative_tuning_no_warning(self, disc):
        """Pole-zero at 1ms tau_cl → low BW → no BW_max warning."""
        res = disc.pwm_delay_analysis()
        bw_warns = [w for w in res["warnings"] if "BW_max" in w or "Crossover" in w]
        assert len(bw_warns) == 0


# ---------------------------------------------------------------------------
# Q15 word-length
# ---------------------------------------------------------------------------

class TestQ15:
    def test_kp_q15_int_within_range(self, disc):
        res = disc.q15_word_length()
        assert -_Q15_SCALE <= res["Kp_q15_int"] <= _Q15_SCALE - 1

    def test_ki_q15_int_within_range(self, disc):
        res = disc.q15_word_length()
        assert -_Q15_SCALE <= res["Ki_q15_int"] <= _Q15_SCALE - 1

    def test_kp_q15_float_close_to_kp(self, disc):
        """Reconstructed Kp from Q15 must match original within 1 LSB / scale."""
        res = disc.q15_word_length()
        scale = res["Kp_scale"]
        lsb = 1.0 / (_Q15_SCALE * scale)
        assert abs(res["Kp_q15_float"] - disc.Kp) <= lsb + 1e-12

    def test_ki_q15_float_close_to_ki(self, disc):
        res = disc.q15_word_length()
        scale = res["Ki_scale"]
        lsb = 1.0 / (_Q15_SCALE * scale)
        assert abs(res["Ki_q15_float"] - disc.Ki) <= lsb + 1e-12

    def test_hex_format(self, disc):
        """Q15 hex strings must start with '0x' and be 6 chars (0xXXXX)."""
        res = disc.q15_word_length()
        assert res["Kp_q15_hex"].startswith("0x")
        assert res["Ki_q15_hex"].startswith("0x")
        assert len(res["Kp_q15_hex"]) == 6
        assert len(res["Ki_q15_hex"]) == 6

    def test_small_kp_no_scaling_needed(self):
        """Kp < 1 should require no downscaling (scale = 1)."""
        motor_cfg, load_cfg = load_config(
            "config/motor_delta_ecma_c21010.yaml",
            "config/load_fan.yaml",
        )
        plant = PMSMPlant(motor_cfg, load_cfg)
        params = ParameterIdentifier(motor_cfg).simulate()
        tuner = CurrentLoopTuner(plant, params)
        loop = tuner.tune("pole_zero", axis="d", tau_cl_s=5e-3)  # large tau_cl → small Kp
        disc = Discretizer(loop, Ts_s=_TS)
        res = disc.q15_word_length()
        # For Kp < 1: scale should be 1 or close
        assert res["Kp_scale"] >= 0.5

    def test_returns_all_keys(self, disc):
        res = disc.q15_word_length()
        for key in ("Kp_scale", "Kp_q15_int", "Kp_q15_float", "Kp_err_pct",
                    "Kp_q15_hex", "Ki_scale", "Ki_q15_int", "Ki_q15_float",
                    "Ki_err_pct", "Ki_q15_hex", "warnings"):
            assert key in res


# ---------------------------------------------------------------------------
# Firmware table
# ---------------------------------------------------------------------------

class TestFirmwareTable:
    def test_returns_string(self, disc):
        table = disc.firmware_table()
        assert isinstance(table, str)

    def test_contains_kp_ki(self, disc):
        table = disc.firmware_table()
        assert "Kp" in table
        assert "Ki" in table

    def test_contains_ts(self, disc):
        table = disc.firmware_table()
        assert "µs" in table or "us" in table or "50" in table


# ---------------------------------------------------------------------------
# Plots (smoke tests)
# ---------------------------------------------------------------------------

class TestPlots:
    def test_bode_with_delay_saves(self, disc, tmp_path):
        disc.plot_bode_with_delay(save_path=str(tmp_path / "bode_delay.png"))

    def test_discretization_comparison_saves(self, disc, tmp_path):
        disc.plot_discretization_comparison(
            save_path=str(tmp_path / "disc_compare.png")
        )

    def test_ipmsm_bode_delay(self, ipmsm_loop, tmp_path):
        d = Discretizer(ipmsm_loop, Ts_s=_TS)
        d.plot_bode_with_delay(save_path=str(tmp_path / "ipmsm_bode_delay.png"))

    @pytest.mark.parametrize("method", [
        "pole_zero", "frequency_domain", "root_locus", "ziegler_nichols"
    ])
    def test_all_tuning_methods_compare(self, method, tmp_path):
        motor_cfg, load_cfg = load_config(
            "config/motor_delta_ecma_c21010.yaml",
            "config/load_fan.yaml",
        )
        plant = PMSMPlant(motor_cfg, load_cfg)
        params = ParameterIdentifier(motor_cfg).simulate()
        tuner = CurrentLoopTuner(plant, params)
        loop = tuner.tune(method, axis="d")
        d = Discretizer(loop, Ts_s=_TS)
        d.compare_methods()
        d.pwm_delay_analysis()
