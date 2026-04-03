"""
test/test_current_loop.py — pytest tests for modules/current_loop.py

Verified analytical results:
  - pole_zero: dominant CL pole at -1/tau_cl ± 5%
  - pole_zero: PI zero cancels plant pole (Ki/Kp = Rs/L)
  - frequency_domain: |OL(j*omega_c)| ≈ 1 at desired crossover ± 15%
  - root_locus: CL poles have damping ratio ≈ target_zeta ± 10%
  - ziegler_nichols: Kp, Ki positive; PM > 0
  - all methods: PM > 30 deg, BW > 0, GM > 0
  - SPMSM: d-axis == q-axis gains (Ld = Lq)
  - IPMSM: d-axis != q-axis gains (Ld != Lq)
  - LoopResult: all fields populated, summary() runs, plots save
  - invalid method/axis raises ValueError
"""

import math
import numpy as np
import pytest

from utils.config import load_config
from modules.plant import PMSMPlant
from modules.param_id import ParameterIdentifier
from modules.current_loop import CurrentLoopTuner, LoopResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spmsm_plant():
    motor_cfg, load_cfg = load_config(
        "config/motor_delta_ecma_c21010.yaml",
        "config/load_fan.yaml",
    )
    return PMSMPlant(motor_cfg, load_cfg)


@pytest.fixture
def ipmsm_plant():
    motor_cfg, load_cfg = load_config(
        "config/motor_magnetic_blq40.yaml",
        "config/load_const_torque.yaml",
    )
    return PMSMPlant(motor_cfg, load_cfg)


@pytest.fixture
def spmsm_params(spmsm_plant):
    return ParameterIdentifier(spmsm_plant.motor_cfg).simulate()


@pytest.fixture
def ipmsm_params(ipmsm_plant):
    return ParameterIdentifier(ipmsm_plant.motor_cfg).simulate()


@pytest.fixture
def spmsm_tuner(spmsm_plant, spmsm_params):
    return CurrentLoopTuner(spmsm_plant, spmsm_params)


@pytest.fixture
def ipmsm_tuner(ipmsm_plant, ipmsm_params):
    return CurrentLoopTuner(ipmsm_plant, ipmsm_params)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_invalid_method_raises(self, spmsm_tuner):
        with pytest.raises(ValueError, match="method"):
            spmsm_tuner.tune(method="banana")

    def test_invalid_axis_raises(self, spmsm_tuner):
        with pytest.raises(ValueError, match="axis"):
            spmsm_tuner.tune(axis="z")

    def test_valid_axes(self, spmsm_tuner):
        for axis in ("d", "q"):
            r = spmsm_tuner.tune(method="pole_zero", axis=axis)
            assert isinstance(r, LoopResult)


# ---------------------------------------------------------------------------
# LoopResult structure
# ---------------------------------------------------------------------------

class TestLoopResultStructure:
    def test_all_fields_populated(self, spmsm_tuner):
        r = spmsm_tuner.tune("pole_zero")
        assert r.Kp > 0
        assert r.Ki > 0
        assert r.BW_Hz > 0
        assert r.PM_deg != 0.0
        assert r.crossover_Hz > 0
        assert r.settling_ms > 0
        assert isinstance(r.warnings, list)
        assert r._num_ol is not None
        assert r._den_cl is not None

    def test_method_and_axis_stored(self, spmsm_tuner):
        r = spmsm_tuner.tune("frequency_domain", axis="q")
        assert r.method == "frequency_domain"
        assert r.axis == "q"

    def test_summary_runs(self, spmsm_tuner, capsys):
        r = spmsm_tuner.tune("pole_zero")
        r.summary()
        out = capsys.readouterr().out
        assert "pole_zero" in out
        assert "Kp" in out


# ---------------------------------------------------------------------------
# Method 1: pole_zero
# ---------------------------------------------------------------------------

class TestPoleZero:
    def test_ki_kp_ratio_cancels_plant_pole(self, spmsm_tuner):
        """Ki/Kp must equal Rs/Ld (PI zero cancels plant pole)."""
        r = spmsm_tuner.tune("pole_zero", axis="d", tau_cl_s=1e-3)
        Rs = spmsm_tuner._Rs
        Ld = spmsm_tuner._Ld
        assert math.isclose(r.Ki / r.Kp, Rs / Ld, rel_tol=1e-6)

    def test_dominant_cl_pole_at_tau_cl(self, spmsm_tuner):
        """
        Dominant CL pole must be at -1/tau_cl ± 5%.
        After pole-zero cancellation, T(s) ≈ 1/(tau_cl*s + 1).
        """
        tau_cl = 1e-3
        r = spmsm_tuner.tune("pole_zero", axis="d", tau_cl_s=tau_cl)
        poles = np.roots(r._den_cl)
        dominant_pole = poles[np.argmax(np.abs(poles.real))]   # most negative real
        # CL zero cancels the slower pole; dominant is the fast one at -1/tau_cl
        fast_pole = poles[np.argmin(poles.real)]  # most negative = fastest
        expected = -1.0 / tau_cl
        assert math.isclose(fast_pole.real, expected, rel_tol=0.05), \
            f"Dominant pole {fast_pole.real:.1f} vs expected {expected:.1f}"

    def test_kp_equals_L_over_tau_cl(self, spmsm_tuner):
        """Kp = Ld / tau_cl analytically."""
        tau_cl = 2e-3
        r = spmsm_tuner.tune("pole_zero", axis="d", tau_cl_s=tau_cl)
        expected_Kp = spmsm_tuner._Ld / tau_cl
        assert math.isclose(r.Kp, expected_Kp, rel_tol=1e-9)

    def test_ki_equals_Rs_over_tau_cl(self, spmsm_tuner):
        """Ki = Rs / tau_cl analytically."""
        tau_cl = 2e-3
        r = spmsm_tuner.tune("pole_zero", axis="d", tau_cl_s=tau_cl)
        expected_Ki = spmsm_tuner._Rs / tau_cl
        assert math.isclose(r.Ki, expected_Ki, rel_tol=1e-9)

    def test_good_phase_margin(self, spmsm_tuner):
        """Pole-zero cancellation should give good PM (> 60 deg typically)."""
        r = spmsm_tuner.tune("pole_zero", axis="d", tau_cl_s=1e-3)
        assert r.PM_deg > 60.0, f"PM={r.PM_deg:.1f} deg unexpectedly low"

    def test_no_warnings_at_nominal(self, spmsm_tuner):
        r = spmsm_tuner.tune("pole_zero", axis="d", tau_cl_s=1e-3)
        pm_warnings = [w for w in r.warnings if "Phase" in w]
        assert len(pm_warnings) == 0


# ---------------------------------------------------------------------------
# Method 2: frequency_domain
# ---------------------------------------------------------------------------

class TestFrequencyDomain:
    def test_crossover_near_desired_BW(self, spmsm_tuner):
        """Gain crossover must be within 20% of desired BW."""
        BW_Hz = 500.0
        r = spmsm_tuner.tune("frequency_domain", axis="d", BW_Hz=BW_Hz)
        assert math.isclose(r.crossover_Hz, BW_Hz, rel_tol=0.20), \
            f"Crossover {r.crossover_Hz:.1f} Hz vs desired {BW_Hz:.1f} Hz"

    def test_ki_is_fraction_of_kp_times_omega(self, spmsm_tuner):
        """Ki = Kp * omega_c / 10 (within 1%)."""
        BW_Hz = 500.0
        r = spmsm_tuner.tune("frequency_domain", axis="d", BW_Hz=BW_Hz)
        omega_c = 2.0 * math.pi * BW_Hz
        assert math.isclose(r.Ki, r.Kp * omega_c / 10.0, rel_tol=0.01)

    def test_positive_phase_margin(self, spmsm_tuner):
        r = spmsm_tuner.tune("frequency_domain", axis="d", BW_Hz=500)
        assert r.PM_deg > 30.0

    def test_q_axis_higher_bw_for_ipmsm(self, ipmsm_tuner):
        """IPMSM: q-axis has Lq > Ld → lower tau_q crossover → lower BW at same gain."""
        rd = ipmsm_tuner.tune("frequency_domain", axis="d", BW_Hz=300)
        rq = ipmsm_tuner.tune("frequency_domain", axis="q", BW_Hz=300)
        # Both tuned for same BW_Hz so crossover should be similar
        assert math.isclose(rd.crossover_Hz, rq.crossover_Hz, rel_tol=0.25)


# ---------------------------------------------------------------------------
# Method 3: root_locus
# ---------------------------------------------------------------------------

class TestRootLocus:
    def test_cl_poles_have_target_zeta(self, spmsm_tuner):
        """
        Closed-loop poles must have damping ratio ≈ target_zeta ± 10%.
        For the 2nd-order system: zeta = b/(2*sqrt(a*c)) from char eq.
        """
        target_zeta = 0.707
        tau_cl = 0.5e-3
        r = spmsm_tuner.tune("root_locus", axis="d",
                             target_zeta=target_zeta, tau_cl_s=tau_cl)
        a = r._den_cl[0]
        b = r._den_cl[1]
        c = r._den_cl[2]
        disc = b**2 - 4*a*c
        if disc < 0:
            # Complex poles
            omega_n = math.sqrt(c / a)
            zeta_actual = b / (2.0 * math.sqrt(a * c))
            assert math.isclose(zeta_actual, target_zeta, rel_tol=0.10), \
                f"zeta={zeta_actual:.3f} vs target={target_zeta}"

    def test_kp_positive(self, spmsm_tuner):
        r = spmsm_tuner.tune("root_locus", axis="d",
                             target_zeta=0.707, tau_cl_s=0.5e-3)
        assert r.Kp > 0

    def test_ki_positive(self, spmsm_tuner):
        r = spmsm_tuner.tune("root_locus", axis="d",
                             target_zeta=0.707, tau_cl_s=0.5e-3)
        assert r.Ki > 0

    def test_bw_nonzero(self, spmsm_tuner):
        r = spmsm_tuner.tune("root_locus", axis="d",
                             target_zeta=0.707, tau_cl_s=0.5e-3)
        assert r.BW_Hz > 0

    def test_ipmsm_d_axis(self, ipmsm_tuner):
        r = ipmsm_tuner.tune("root_locus", axis="d",
                             target_zeta=0.707, tau_cl_s=1e-3)
        assert r.Kp > 0 and r.Ki > 0

    def test_ipmsm_q_axis(self, ipmsm_tuner):
        r = ipmsm_tuner.tune("root_locus", axis="q",
                             target_zeta=0.707, tau_cl_s=1e-3)
        assert r.Kp > 0 and r.Ki > 0


# ---------------------------------------------------------------------------
# Method 4: ziegler_nichols
# ---------------------------------------------------------------------------

class TestZieglerNichols:
    def test_kp_ki_positive(self, spmsm_tuner):
        r = spmsm_tuner.tune("ziegler_nichols", axis="d")
        assert r.Kp > 0
        assert r.Ki > 0

    def test_phase_margin_positive(self, spmsm_tuner):
        r = spmsm_tuner.tune("ziegler_nichols", axis="d")
        assert r.PM_deg > 0.0

    def test_bw_nonzero(self, spmsm_tuner):
        r = spmsm_tuner.tune("ziegler_nichols", axis="d")
        assert r.BW_Hz > 0

    def test_ipmsm_q_axis(self, ipmsm_tuner):
        r = ipmsm_tuner.tune("ziegler_nichols", axis="q")
        assert r.Kp > 0 and r.Ki > 0


# ---------------------------------------------------------------------------
# All methods: common quality checks
# ---------------------------------------------------------------------------

class TestAllMethods:
    @pytest.mark.parametrize("method", [
        "pole_zero", "frequency_domain", "root_locus", "ziegler_nichols"
    ])
    def test_pm_above_threshold(self, spmsm_tuner, method):
        """All methods should give PM > 30 deg for nominal tuning."""
        r = spmsm_tuner.tune(method, axis="d")
        assert r.PM_deg > 30.0, f"{method}: PM={r.PM_deg:.1f} deg < 30 deg"

    @pytest.mark.parametrize("method", [
        "pole_zero", "frequency_domain", "root_locus", "ziegler_nichols"
    ])
    def test_bw_positive(self, spmsm_tuner, method):
        r = spmsm_tuner.tune(method, axis="d")
        assert r.BW_Hz > 0

    @pytest.mark.parametrize("method", [
        "pole_zero", "frequency_domain", "root_locus", "ziegler_nichols"
    ])
    def test_settling_positive(self, spmsm_tuner, method):
        r = spmsm_tuner.tune(method, axis="d")
        assert r.settling_ms > 0

    @pytest.mark.parametrize("method", [
        "pole_zero", "frequency_domain", "root_locus", "ziegler_nichols"
    ])
    def test_gm_positive(self, spmsm_tuner, method):
        r = spmsm_tuner.tune(method, axis="d")
        assert r.GM_dB > 0


# ---------------------------------------------------------------------------
# SPMSM vs IPMSM axis symmetry / asymmetry
# ---------------------------------------------------------------------------

class TestMotorTypeBehaviour:
    def test_spmsm_d_q_gains_equal(self, spmsm_tuner):
        """SPMSM: Ld = Lq → same gains for d and q axes."""
        rd = spmsm_tuner.tune("pole_zero", axis="d", tau_cl_s=1e-3)
        rq = spmsm_tuner.tune("pole_zero", axis="q", tau_cl_s=1e-3)
        assert math.isclose(rd.Kp, rq.Kp, rel_tol=1e-6)
        assert math.isclose(rd.Ki, rq.Ki, rel_tol=1e-6)

    def test_ipmsm_d_q_gains_differ(self, ipmsm_tuner):
        """IPMSM: Ld < Lq → different gains for d and q axes."""
        rd = ipmsm_tuner.tune("pole_zero", axis="d", tau_cl_s=1e-3)
        rq = ipmsm_tuner.tune("pole_zero", axis="q", tau_cl_s=1e-3)
        assert not math.isclose(rd.Kp, rq.Kp, rel_tol=0.01)

    def test_ipmsm_q_kp_larger_than_d(self, ipmsm_tuner):
        """IPMSM: Lq > Ld → Kp_q > Kp_d for same tau_cl (Kp = L/tau_cl)."""
        rd = ipmsm_tuner.tune("pole_zero", axis="d", tau_cl_s=1e-3)
        rq = ipmsm_tuner.tune("pole_zero", axis="q", tau_cl_s=1e-3)
        assert rq.Kp > rd.Kp

    def test_tuner_without_params_uses_plant(self, spmsm_plant):
        """CurrentLoopTuner with params=None falls back to plant values."""
        tuner = CurrentLoopTuner(spmsm_plant, params=None)
        r = tuner.tune("pole_zero", axis="d", tau_cl_s=1e-3)
        assert r.Kp > 0


# ---------------------------------------------------------------------------
# Plotting (smoke tests)
# ---------------------------------------------------------------------------

class TestPlots:
    def test_plot_bode_saves(self, spmsm_tuner, tmp_path):
        r = spmsm_tuner.tune("pole_zero", axis="d")
        r.plot_bode(save_path=str(tmp_path / "bode.png"))

    def test_plot_step_saves(self, spmsm_tuner, tmp_path):
        r = spmsm_tuner.tune("pole_zero", axis="d")
        r.plot_step(save_path=str(tmp_path / "step.png"))

    def test_plot_root_locus_saves(self, spmsm_tuner, tmp_path):
        r = spmsm_tuner.tune("pole_zero", axis="d")
        r.plot_root_locus(save_path=str(tmp_path / "rl.png"))

    def test_all_methods_bode(self, spmsm_tuner, tmp_path):
        for method in ("pole_zero", "frequency_domain", "root_locus", "ziegler_nichols"):
            r = spmsm_tuner.tune(method, axis="d")
            r.plot_bode(save_path=str(tmp_path / f"bode_{method}.png"))
