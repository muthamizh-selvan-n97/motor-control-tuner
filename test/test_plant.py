"""
test/test_plant.py — pytest tests for modules/plant.py

Verified analytical results:
  - State-space poles match -Rs/Ld and -Rs/Lq
  - Current plant TF DC gain = 1/Rs
  - Speed plant TF matches Kt_eff / (J_total*s + B_total)
  - SPMSM: mtpa_angle returns 0, Lq forced = Ld
  - IPMSM: mtpa_currents satisfy id^2 + iq^2 = Is^2
  - Electromagnetic torque: SPMSM reluctance term is zero
  - Discretization: ZOH eigenvalues match exp(lambda * Ts)
  - Field weakening: id_fw <= 0, clamped to [-Is_max, 0]
  - Mechanical derivative: at steady state Te = TL + B*omega
"""

import math

import numpy as np
import pytest

from utils.config import load_config
from modules.plant import PMSMPlant

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def spmsm_fan():
    motor_cfg, load_cfg = load_config(
        "config/motor_delta_ecma_c21010.yaml",
        "config/load_fan.yaml",
    )
    return PMSMPlant(motor_cfg, load_cfg)


@pytest.fixture
def ipmsm_const():
    motor_cfg, load_cfg = load_config(
        "config/motor_magnetic_blq40.yaml",
        "config/load_const_torque.yaml",
    )
    return PMSMPlant(motor_cfg, load_cfg)


@pytest.fixture
def spmsm_servo():
    motor_cfg, load_cfg = load_config(
        "config/motor_delta_ecma_c21010.yaml",
        "config/load_position_servo.yaml",
    )
    return PMSMPlant(motor_cfg, load_cfg)


# ---------------------------------------------------------------------------
# State-space structure
# ---------------------------------------------------------------------------

class TestStateSpace:
    def test_A_poles_match_electrical_time_constants(self, spmsm_fan):
        """A matrix eigenvalues must equal -Rs/Ld and -Rs/Lq."""
        p = spmsm_fan
        A_c, _, _ = p.build_state_space()
        eigenvalues = np.linalg.eigvals(A_c)
        expected = sorted([-p.Rs / p.Ld, -p.Rs / p.Lq])
        actual = sorted(eigenvalues.real.tolist())
        for e, a in zip(expected, actual):
            assert math.isclose(e, a, rel_tol=1e-9), f"Expected pole {e}, got {a}"

    def test_A_is_diagonal(self, spmsm_fan):
        """Decoupled model: A must be diagonal (cross-coupling excluded)."""
        A_c, _, _ = spmsm_fan.build_state_space()
        assert A_c[0, 1] == 0.0
        assert A_c[1, 0] == 0.0

    def test_B_diagonal_entries(self, ipmsm_const):
        """B[0,0]=1/Ld, B[1,1]=1/Lq."""
        p = ipmsm_const
        _, B_c, _ = p.build_state_space()
        assert math.isclose(B_c[0, 0], 1.0 / p.Ld, rel_tol=1e-9)
        assert math.isclose(B_c[1, 1], 1.0 / p.Lq, rel_tol=1e-9)

    def test_C_is_identity(self, spmsm_fan):
        _, _, C_c = spmsm_fan.build_state_space()
        np.testing.assert_array_equal(C_c, np.eye(2))

    def test_ipmsm_d_q_poles_differ(self, ipmsm_const):
        """IPMSM Ld != Lq → two distinct poles."""
        p = ipmsm_const
        A_c, _, _ = p.build_state_space()
        eigs = np.linalg.eigvals(A_c).real
        assert not math.isclose(eigs[0], eigs[1], rel_tol=0.01)


# ---------------------------------------------------------------------------
# Transfer function
# ---------------------------------------------------------------------------

class TestTransferFunction:
    def test_d_axis_dc_gain(self, spmsm_fan):
        """DC gain of G_d(s) at s=0 is 1/Rs."""
        p = spmsm_fan
        num, den = p.get_current_plant_tf('d')
        dc_gain = num[0] / den[-1]   # num[0]/den[1] since den=[tau,1]
        assert math.isclose(dc_gain, 1.0 / p.Rs, rel_tol=1e-9)

    def test_q_axis_dc_gain(self, ipmsm_const):
        """DC gain of G_q(s) at s=0 is 1/Rs."""
        p = ipmsm_const
        num, den = p.get_current_plant_tf('q')
        dc_gain = num[0] / den[-1]
        assert math.isclose(dc_gain, 1.0 / p.Rs, rel_tol=1e-9)

    def test_d_axis_time_constant(self, spmsm_fan):
        """den[0]/den[1] of G_d = tau_d = Ld/Rs."""
        p = spmsm_fan
        num, den = p.get_current_plant_tf('d')
        tau = den[0] / den[1]
        assert math.isclose(tau, p.Ld / p.Rs, rel_tol=1e-9)

    def test_q_axis_time_constant(self, ipmsm_const):
        p = ipmsm_const
        num, den = p.get_current_plant_tf('q')
        tau = den[0] / den[1]
        assert math.isclose(tau, p.Lq / p.Rs, rel_tol=1e-9)

    def test_invalid_axis_raises(self, spmsm_fan):
        with pytest.raises(ValueError):
            spmsm_fan.get_current_plant_tf('x')

    def test_speed_plant_numerator(self, spmsm_fan):
        """Speed plant num = Kt_eff = 1.5*p*psi_f."""
        p = spmsm_fan
        num, den = p.get_speed_plant_tf()
        Kt_eff = 1.5 * p.p * p.psi_f
        assert math.isclose(num[0], Kt_eff, rel_tol=1e-9)

    def test_speed_plant_denominator(self, spmsm_fan):
        """Speed plant den = [J_total, B_total]."""
        p = spmsm_fan
        num, den = p.get_speed_plant_tf()
        assert math.isclose(den[0], p.J_total, rel_tol=1e-9)
        assert math.isclose(den[1], p.B_total, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# SPMSM-specific behaviour
# ---------------------------------------------------------------------------

class TestSPMSM:
    def test_lq_equals_ld(self, spmsm_fan):
        """SPMSM must enforce Lq = Ld."""
        p = spmsm_fan
        assert p.Ld == p.Lq

    def test_mtpa_angle_is_zero(self, spmsm_fan):
        """SPMSM has no reluctance torque → beta = 0."""
        assert spmsm_fan.mtpa_angle(7.3) == 0.0

    def test_mtpa_currents_spmsm(self, spmsm_fan):
        """SPMSM: id* = 0, iq* = Is."""
        id_star, iq_star = spmsm_fan.mtpa_currents(7.3)
        assert id_star == 0.0
        assert math.isclose(iq_star, 7.3, rel_tol=1e-9)

    def test_torque_no_reluctance_term(self, spmsm_fan):
        """For SPMSM: Te = 1.5*p*psi_f*iq regardless of id."""
        p = spmsm_fan
        id, iq = 2.0, 5.0
        Te = p.electromagnetic_torque(id, iq)
        Te_expected = 1.5 * p.p * p.psi_f * iq
        assert math.isclose(Te, Te_expected, rel_tol=1e-9)

    def test_j_total(self, spmsm_fan):
        """J_total = J_motor + J_load."""
        p = spmsm_fan
        assert math.isclose(
            p.J_total, p.J_motor + p.J_load, rel_tol=1e-9
        )

    def test_b_total(self, spmsm_fan):
        """B_total = B_motor + B_load."""
        p = spmsm_fan
        assert math.isclose(
            p.B_total, p.B_motor + p.B_load, rel_tol=1e-9
        )


# ---------------------------------------------------------------------------
# IPMSM-specific behaviour
# ---------------------------------------------------------------------------

class TestIPMSM:
    def test_lq_greater_than_ld(self, ipmsm_const):
        assert ipmsm_const.Lq > ipmsm_const.Ld

    def test_mtpa_angle_positive(self, ipmsm_const):
        """IPMSM beta > 0 for Is > 0."""
        beta = ipmsm_const.mtpa_angle(5.6)
        assert beta > 0.0

    def test_mtpa_currents_magnitude(self, ipmsm_const):
        """id*^2 + iq*^2 must equal Is^2 (current vector on circle)."""
        Is = 5.6
        id_star, iq_star = ipmsm_const.mtpa_currents(Is)
        assert math.isclose(id_star**2 + iq_star**2, Is**2, rel_tol=1e-6)

    def test_mtpa_id_negative(self, ipmsm_const):
        """IPMSM MTPA: id* < 0 (flux weakening component)."""
        id_star, _ = ipmsm_const.mtpa_currents(5.6)
        assert id_star < 0.0

    def test_mtpa_zero_current(self, ipmsm_const):
        """At Is=0 no crash, returns (0, 0)."""
        id_star, iq_star = ipmsm_const.mtpa_currents(0.0)
        assert id_star == 0.0
        assert iq_star == 0.0

    def test_torque_includes_reluctance(self, ipmsm_const):
        """IPMSM torque > SPMSM-equivalent torque due to reluctance term."""
        p = ipmsm_const
        id, iq = -1.0, 5.0
        Te = p.electromagnetic_torque(id, iq)
        Te_magnet_only = 1.5 * p.p * p.psi_f * iq
        # (Ld - Lq)*id*iq: Ld<Lq, id<0, iq>0 → positive reluctance contribution
        assert Te > Te_magnet_only


# ---------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------

class TestDiscretization:
    @pytest.mark.parametrize("method", ["zoh", "tustin"])
    def test_discrete_eigenvalues(self, spmsm_fan, method):
        """
        ZOH/Tustin discrete eigenvalues must approximate exp(lambda_c * Ts).
        Tolerance 5% — Tustin introduces bilinear warping.
        """
        Ts = 50e-6
        p = spmsm_fan
        A_c, _, _ = p.build_state_space()
        lambda_c = np.linalg.eigvals(A_c).real

        A_d, _ = p.discretize(Ts, method)
        lambda_d = np.linalg.eigvals(A_d).real

        expected = np.sort(np.exp(lambda_c * Ts))
        actual = np.sort(lambda_d)
        np.testing.assert_allclose(actual, expected, rtol=0.05)

    def test_zoh_stable(self, spmsm_fan):
        """All discrete eigenvalues must be inside the unit circle."""
        A_d, _ = spmsm_fan.discretize(50e-6, "zoh")
        eigs = np.abs(np.linalg.eigvals(A_d))
        assert np.all(eigs < 1.0)

    def test_invalid_ts_raises(self, spmsm_fan):
        with pytest.raises(ValueError):
            spmsm_fan.discretize(-1e-6)

    def test_invalid_method_raises(self, spmsm_fan):
        with pytest.raises(ValueError):
            spmsm_fan.discretize(50e-6, method="euler")


# ---------------------------------------------------------------------------
# Field weakening
# ---------------------------------------------------------------------------

class TestFieldWeakening:
    def test_id_fw_negative_or_zero(self, spmsm_fan):
        """Field weakening id* must be <= 0."""
        p = spmsm_fan
        omega_e = p.p * 5000 * math.pi / 30   # above rated
        id_fw = p.field_weakening_id(omega_e, Vdc=300.0, Is_max=21.9)
        assert id_fw <= 0.0

    def test_id_fw_zero_at_zero_speed(self, spmsm_fan):
        """At omega_e = 0, no field weakening needed."""
        id_fw = spmsm_fan.field_weakening_id(0.0, Vdc=300.0, Is_max=21.9)
        assert id_fw == 0.0

    def test_id_fw_clamped_to_Is_max(self, spmsm_fan):
        """id_fw must not exceed -Is_max in magnitude."""
        p = spmsm_fan
        omega_e = p.p * 100000 * math.pi / 30   # extreme over-speed
        Is_max = 21.9
        id_fw = p.field_weakening_id(omega_e, Vdc=300.0, Is_max=Is_max)
        assert id_fw >= -Is_max


# ---------------------------------------------------------------------------
# Load torque and mechanical dynamics
# ---------------------------------------------------------------------------

class TestLoadAndMechanics:
    def test_fan_torque_quadratic(self, spmsm_fan):
        """Fan TL = k_fan * omega^2."""
        p = spmsm_fan
        omega = 314.2
        TL = p.load_torque(omega)
        assert math.isclose(TL, p.k_fan * omega**2, rel_tol=1e-9)

    def test_const_torque_load(self, ipmsm_const):
        """Constant torque load returns TL_const regardless of speed."""
        p = ipmsm_const
        assert p.load_torque(0.0) == p.TL_const
        assert p.load_torque(314.2) == p.TL_const

    def test_position_servo_zero_load(self, spmsm_servo):
        """Position servo TL = 0."""
        assert spmsm_servo.load_torque(314.2) == 0.0

    def test_mechanical_steady_state(self, spmsm_fan):
        """
        At steady state d(omega)/dt = 0:
            Te = TL + B_total * omega_ss
        Verify mechanical_derivative returns ~0 when Te = TL + B*omega.
        """
        p = spmsm_fan
        omega_ss = 314.2
        TL = p.load_torque(omega_ss)
        Te_needed = TL + p.B_total * omega_ss
        # Find iq that produces Te_needed (id=0, SPMSM)
        iq_ss = Te_needed / (1.5 * p.p * p.psi_f)
        domega_dt = p.mechanical_derivative(omega_ss, id=0.0, iq=iq_ss)
        assert math.isclose(domega_dt, 0.0, abs_tol=1e-9)
