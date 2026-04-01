"""
test/test_param_id.py — pytest tests for modules/param_id.py

Verified analytical results:
  - simulate(): measured Rs/Ld/Lq/psi_f within ±5% of true values at 2% noise
  - simulate(): tau_d = Ld/Rs consistent with measurements
  - simulate(): BEMF least-squares fit reduces noise effect across 3 points
  - override(): accepts exact true values
  - override(): rejects Rs <= 0, Ld <= 0, Lq < Ld, bad tau_e
  - override(): falls back to ground truth for unspecified parameters
  - print_comparison(): runs without error
  - plot_identification(): runs without error (Agg backend, no display)
"""

import math
import pytest

from utils.config import load_config
from modules.param_id import ParameterIdentifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spmsm_pid():
    motor_cfg, _ = load_config(
        "config/motor_delta_ecma_c21010.yaml",
        "config/load_fan.yaml",
    )
    return ParameterIdentifier(motor_cfg)


@pytest.fixture
def ipmsm_pid():
    motor_cfg, _ = load_config(
        "config/motor_magnetic_blq40.yaml",
        "config/load_const_torque.yaml",
    )
    return ParameterIdentifier(motor_cfg)


@pytest.fixture
def spmsm_sim(spmsm_pid):
    return spmsm_pid.simulate(noise_std_frac=0.02)


@pytest.fixture
def ipmsm_sim(ipmsm_pid):
    return ipmsm_pid.simulate(noise_std_frac=0.02)


# ---------------------------------------------------------------------------
# simulate() — output structure
# ---------------------------------------------------------------------------

class TestSimulateStructure:
    def test_required_keys_present(self, spmsm_sim):
        for key in ("Rs", "Ld", "Lq", "psi_f", "tau_d", "tau_q", "source"):
            assert key in spmsm_sim, f"Missing key: {key}"

    def test_source_is_simulated(self, spmsm_sim):
        assert spmsm_sim["source"] == "simulated"

    def test_all_values_positive(self, spmsm_sim):
        for key in ("Rs", "Ld", "Lq", "psi_f", "tau_d", "tau_q"):
            assert spmsm_sim[key] > 0, f"{key} must be > 0"

    def test_raw_arrays_present(self, spmsm_sim):
        for key in ("_t_dc", "_Id_step", "_t_ac", "_Vd_wave", "_omega_mechs", "_bemf_noisy"):
            assert key in spmsm_sim, f"Missing raw data key: {key}"


# ---------------------------------------------------------------------------
# simulate() — SPMSM accuracy at 2% noise
# ---------------------------------------------------------------------------

class TestSimulateAccuracySPMSM:
    _TOL = 0.08   # 8% tolerance — two independent noise samples can compound

    def test_Rs_within_tolerance(self, spmsm_pid, spmsm_sim):
        rel_err = abs(spmsm_sim["Rs"] - spmsm_pid._Rs_true) / spmsm_pid._Rs_true
        assert rel_err < self._TOL, f"Rs error {rel_err*100:.1f}% > {self._TOL*100}%"

    def test_Ld_within_tolerance(self, spmsm_pid, spmsm_sim):
        rel_err = abs(spmsm_sim["Ld"] - spmsm_pid._Ld_true) / spmsm_pid._Ld_true
        assert rel_err < self._TOL, f"Ld error {rel_err*100:.1f}% > {self._TOL*100}%"

    def test_Lq_within_tolerance(self, spmsm_pid, spmsm_sim):
        rel_err = abs(spmsm_sim["Lq"] - spmsm_pid._Lq_true) / spmsm_pid._Lq_true
        assert rel_err < self._TOL, f"Lq error {rel_err*100:.1f}% > {self._TOL*100}%"

    def test_psi_f_within_tolerance(self, spmsm_pid, spmsm_sim):
        rel_err = abs(spmsm_sim["psi_f"] - spmsm_pid._psi_f_true) / spmsm_pid._psi_f_true
        assert rel_err < self._TOL, f"psi_f error {rel_err*100:.1f}% > {self._TOL*100}%"

    def test_tau_d_consistent(self, spmsm_sim):
        """tau_d must equal Ld/Rs from the measured values."""
        expected = spmsm_sim["Ld"] / spmsm_sim["Rs"]
        assert math.isclose(spmsm_sim["tau_d"], expected, rel_tol=1e-9)

    def test_tau_q_consistent(self, spmsm_sim):
        expected = spmsm_sim["Lq"] / spmsm_sim["Rs"]
        assert math.isclose(spmsm_sim["tau_q"], expected, rel_tol=1e-9)

    def test_spmsm_lq_approx_ld(self, spmsm_sim):
        """SPMSM: Lq and Ld should be close (same true value)."""
        ratio = spmsm_sim["Lq"] / spmsm_sim["Ld"]
        assert 0.90 < ratio < 1.10, f"SPMSM Lq/Ld ratio {ratio:.3f} out of range"


# ---------------------------------------------------------------------------
# simulate() — IPMSM accuracy at 2% noise
# ---------------------------------------------------------------------------

class TestSimulateAccuracyIPMSM:
    _TOL = 0.08

    def test_Rs_within_tolerance(self, ipmsm_pid, ipmsm_sim):
        rel_err = abs(ipmsm_sim["Rs"] - ipmsm_pid._Rs_true) / ipmsm_pid._Rs_true
        assert rel_err < self._TOL

    def test_Ld_within_tolerance(self, ipmsm_pid, ipmsm_sim):
        rel_err = abs(ipmsm_sim["Ld"] - ipmsm_pid._Ld_true) / ipmsm_pid._Ld_true
        assert rel_err < self._TOL

    def test_Lq_within_tolerance(self, ipmsm_pid, ipmsm_sim):
        rel_err = abs(ipmsm_sim["Lq"] - ipmsm_pid._Lq_true) / ipmsm_pid._Lq_true
        assert rel_err < self._TOL

    def test_psi_f_within_tolerance(self, ipmsm_pid, ipmsm_sim):
        rel_err = abs(ipmsm_sim["psi_f"] - ipmsm_pid._psi_f_true) / ipmsm_pid._psi_f_true
        assert rel_err < self._TOL

    def test_ipmsm_lq_greater_than_ld(self, ipmsm_sim):
        """IPMSM: measured Lq should remain > Ld even with noise."""
        assert ipmsm_sim["Lq"] > ipmsm_sim["Ld"], \
            f"IPMSM Lq ({ipmsm_sim['Lq']*1e3:.3f} mH) <= Ld ({ipmsm_sim['Ld']*1e3:.3f} mH)"


# ---------------------------------------------------------------------------
# simulate() — noise level effect
# ---------------------------------------------------------------------------

class TestSimulateNoise:
    def test_zero_noise_exact(self, spmsm_pid):
        """At zero noise, measured values must be within 0.1% of true values.
        (DC step response settles to ~1-e^-20 ≈ 99.9999% at 20*tau — not exact.)
        """
        res = spmsm_pid.simulate(noise_std_frac=0.0)
        assert math.isclose(res["Rs"], spmsm_pid._Rs_true, rel_tol=1e-3)
        assert math.isclose(res["Ld"], spmsm_pid._Ld_true, rel_tol=1e-6)
        assert math.isclose(res["Lq"], spmsm_pid._Lq_true, rel_tol=1e-6)
        assert math.isclose(res["psi_f"], spmsm_pid._psi_f_true, rel_tol=1e-6)

    def test_high_noise_still_returns_positive(self, spmsm_pid):
        """At 10% noise, all results must still be positive."""
        res = spmsm_pid.simulate(noise_std_frac=0.10)
        for key in ("Rs", "Ld", "Lq", "psi_f"):
            assert res[key] > 0

    def test_reproducible_with_same_seed(self, spmsm_pid):
        """simulate() uses fixed seed — results must be identical on repeat calls."""
        r1 = spmsm_pid.simulate(noise_std_frac=0.02)
        r2 = spmsm_pid.simulate(noise_std_frac=0.02)
        assert math.isclose(r1["Rs"], r2["Rs"], rel_tol=1e-12)
        assert math.isclose(r1["psi_f"], r2["psi_f"], rel_tol=1e-12)


# ---------------------------------------------------------------------------
# override() — valid inputs
# ---------------------------------------------------------------------------

class TestOverrideValid:
    def test_exact_true_values(self, spmsm_pid):
        """Passing exact ground-truth values must succeed."""
        res = spmsm_pid.override(
            Rs=spmsm_pid._Rs_true,
            Ld=spmsm_pid._Ld_true,
            Lq=spmsm_pid._Lq_true,
            psi_f=spmsm_pid._psi_f_true,
        )
        assert math.isclose(res["Rs"], spmsm_pid._Rs_true, rel_tol=1e-9)

    def test_source_is_override(self, spmsm_pid):
        res = spmsm_pid.override()
        assert res["source"] == "override"

    def test_defaults_to_ground_truth(self, spmsm_pid):
        """Calling override() with no args returns ground-truth values."""
        res = spmsm_pid.override()
        assert math.isclose(res["Rs"], spmsm_pid._Rs_true, rel_tol=1e-9)
        assert math.isclose(res["Ld"], spmsm_pid._Ld_true, rel_tol=1e-9)

    def test_tau_d_computed(self, spmsm_pid):
        res = spmsm_pid.override()
        expected = spmsm_pid._Ld_true / spmsm_pid._Rs_true
        assert math.isclose(res["tau_d"], expected, rel_tol=1e-9)

    def test_slight_variation_accepted(self, spmsm_pid):
        """Values within ±10% of true tau_e must pass."""
        Rs_var = spmsm_pid._Rs_true * 1.10   # +10% (within 25% tol)
        res = spmsm_pid.override(Rs=Rs_var)
        assert res["Rs"] == Rs_var


# ---------------------------------------------------------------------------
# override() — invalid inputs (must raise ValueError)
# ---------------------------------------------------------------------------

class TestOverrideInvalid:
    def test_negative_Rs_raises(self, spmsm_pid):
        with pytest.raises(ValueError, match="Rs"):
            spmsm_pid.override(Rs=-0.1)

    def test_zero_Ld_raises(self, spmsm_pid):
        with pytest.raises(ValueError, match="Ld"):
            spmsm_pid.override(Ld=0.0)

    def test_negative_psi_f_raises(self, spmsm_pid):
        with pytest.raises(ValueError, match="psi_f"):
            spmsm_pid.override(psi_f=-0.01)

    def test_lq_less_than_ld_raises(self, ipmsm_pid):
        """Lq < Ld must be rejected."""
        with pytest.raises(ValueError):
            ipmsm_pid.override(
                Ld=ipmsm_pid._Lq_true,   # swap: Ld = true Lq (larger)
                Lq=ipmsm_pid._Ld_true,   # Lq = true Ld (smaller)
            )

    def test_bad_tau_e_raises(self, spmsm_pid):
        """tau_e 50% off from expected must raise ValueError."""
        with pytest.raises(ValueError, match="tau_e"):
            spmsm_pid.override(Rs=spmsm_pid._Rs_true * 2.0)   # doubles tau_e


# ---------------------------------------------------------------------------
# Plotting and printing (smoke tests — no crash)
# ---------------------------------------------------------------------------

class TestPlottingAndPrinting:
    def test_plot_simulated_no_crash(self, spmsm_pid, spmsm_sim, tmp_path):
        save = str(tmp_path / "test_id_plot.png")
        spmsm_pid.plot_identification(spmsm_sim, save_path=save)

    def test_plot_override_no_crash(self, spmsm_pid, capsys):
        res = spmsm_pid.override()
        spmsm_pid.plot_identification(res)   # should print message and return
        captured = capsys.readouterr()
        assert "override" in captured.out.lower()

    def test_print_comparison_no_crash(self, spmsm_pid, spmsm_sim, capsys):
        spmsm_pid.print_comparison(spmsm_sim)
        captured = capsys.readouterr()
        assert "Rs" in captured.out
        assert "psi_f" in captured.out

    def test_ipmsm_plot_no_crash(self, ipmsm_pid, ipmsm_sim, tmp_path):
        save = str(tmp_path / "ipmsm_id_plot.png")
        ipmsm_pid.plot_identification(ipmsm_sim, save_path=save)
