"""
modules/param_id.py — Simulated parameter identification for PMSM.

Two-path design:
  1. simulate() — inject test signals with Gaussian noise, extract Rs/Ld/Lq/psi_f
  2. override() — accept measured values, validate physical consistency

Both paths return the same dict:
  {'Rs': float, 'Ld': float, 'Lq': float, 'psi_f': float,
   'tau_d': float, 'tau_q': float, 'source': str}

Mechanical identification (requires load_cfg at construction):
  3. simulate_mechanical() — four additional bench tests
  4. override()            — also accepts p, KE_SI, J_total, J_load, B_total, B_load, ...

Bench tests simulated:
  Test 1 — Rs    : DC lockout step, measure steady-state Id
  Test 2 — Ld,Lq : AC standstill injection at 100 Hz, two rotor positions
  Test 3 — psi_f : BEMF open-circuit measurement at three speeds
  Test 4 — p     : electrical frequency counting from BEMF zero-crossings
  Test 5 — KE_SI : slope of BEMF vs omega_mech (V·s/rad)
  Test 6 — J_total : no-load acceleration step, linear ramp fit
  Test 7 — B_total + load coefficients : steady-state torque sweep
"""

import math

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Named constants — electrical tests
# ---------------------------------------------------------------------------
_AC_INJECTION_FREQ_HZ = 100.0             # standstill injection frequency
_BEMF_SPEEDS_RPM = (1000.0, 2000.0, 3000.0)   # open-circuit BEMF test speeds
_TAU_E_TOLERANCE = 0.25                   # ±25% tolerance on tau_e consistency check
_NOISE_DEFAULT = 0.02                     # 2% default noise std fraction
_MIN_SALIENCY = 1.0                       # Lq/Ld must be >= 1.0

# ---------------------------------------------------------------------------
# Named constants — mechanical tests
# ---------------------------------------------------------------------------
_POLE_PAIRS_TEST_RPM   = 1000.0   # mechanical speed for pole-pairs electrical-freq test
_ACCEL_TEST_IQ_A       = 1.0      # iq injected during no-load acceleration (A)
_ACCEL_TEST_T_S        = 2.0      # duration of acceleration window (s)
_ACCEL_TEST_N_SAMPLES  = 500      # number of speed samples during acceleration
_DAMPING_TEST_N_SPEEDS = 10       # number of steady-state speed points for B/k_fan fit
_DAMPING_TEST_FRAC_MIN = 0.10     # lowest test speed as fraction of rated speed
_DAMPING_TEST_FRAC_MAX = 1.00     # highest test speed as fraction of rated speed
_J_CONSISTENCY_TOL     = 0.01     # 1% tolerance for J_total/J_load consistency check
_J_MOTOR_MIN_FRAC      = 1.0      # J_total must be >= J_motor_true (no subtraction below zero)


class ParameterIdentifier:
    """
    Simulate or accept measured PMSM electrical and mechanical parameters.

    Parameters
    ----------
    motor_cfg : dict
        Validated motor config from utils.config.load_config().
        Used as the ground-truth plant for simulated tests.
    load_cfg : dict, optional
        Validated load config from utils.config.load_config().
        Required for simulate_mechanical() and mechanical override() calls.
    """

    def __init__(self, motor_cfg: dict, load_cfg: dict = None) -> None:
        self.motor_cfg = motor_cfg
        self.load_cfg = load_cfg
        self.motor_type: str = motor_cfg["motor_type"]

        # Ground-truth electrical parameters (used only by simulate())
        elec = motor_cfg["electrical"]
        self._Rs_true: float = elec["Rs_ohm"]
        self._Ld_true: float = elec["Ld_H"]
        self._Lq_true: float = elec["Lq_H"]
        self._psi_f_true: float = elec["psi_f_Wb"]
        self._p_true: int = elec["pole_pairs"]

        # Ground-truth mechanical parameters (only when load_cfg provided)
        if load_cfg is not None:
            mech = motor_cfg["mechanical"]
            self._J_motor_true: float = mech["J_kgm2"]
            self._B_motor_true: float = mech["B_Nms_rad"]
            self._J_load_true: float  = load_cfg["J_load_kgm2"]
            self._B_load_true: float  = load_cfg["B_load_Nms_rad"]
            self._J_total_true: float = self._J_motor_true + self._J_load_true
            self._B_total_true: float = self._B_motor_true + self._B_load_true
            self._load_type: str      = load_cfg["load_type"]
            self._k_fan_true: float   = load_cfg.get("k_fan", 0.0)
            self._TL_const_true: float = load_cfg.get("TL_Nm", 0.0)
            self._omega_rated: float  = motor_cfg["rated"]["speed_rad_s"]

        # Storage for last simulation run (for plotting)
        self._last_sim: dict = {}

    # ------------------------------------------------------------------
    # Public API — electrical
    # ------------------------------------------------------------------

    def simulate(self, noise_std_frac: float = _NOISE_DEFAULT) -> dict:
        """
        Simulate three bench tests with Gaussian measurement noise.

        Test 1 — Rs (DC lockout test):
            Apply Vdc step at d-axis, wait for steady state (id = Vdc/Rs).
            Rs_meas = Vdc / Id_ss
            Noise: ±noise_std_frac on both voltage and current.

        Test 2 — Ld, Lq (AC standstill injection at 100 Hz):
            Inject sinusoidal voltage at two rotor positions (0°, 90° elec.).
            Ld = |Vd_peak / (omega_inj * Id_peak)|   (d-axis locked)
            Lq = |Vq_peak / (omega_inj * Iq_peak)|   (q-axis locked)
            Noise: ±noise_std_frac on voltage and current amplitudes.

        Test 3 — psi_f (BEMF open-circuit):
            Spin at three speeds. Measure peak phase BEMF.
            BEMF_peak = psi_f * p * omega_mech
            Fit with least-squares: psi_f = sum(BEMF_i * omega_i) / sum(omega_i^2)

        Parameters
        ----------
        noise_std_frac : float
            Gaussian noise standard deviation as a fraction of the true value.
            Default 0.02 (2%).

        Returns
        -------
        dict with keys: Rs, Ld, Lq, psi_f, tau_d, tau_q, source,
                        plus raw measurement arrays for plotting.
        """
        rng = np.random.default_rng(seed=42)   # reproducible noise

        def noisy(value: float) -> float:
            return value * (1.0 + rng.normal(0.0, noise_std_frac))

        # ------ Test 1: Rs -----------------------------------------------
        Vdc_step = self._Rs_true * 10.0   # choose step so Id ≈ 10× rated noise
        # Simulate RC-like step response: sample at t >> tau_e
        t_dc = np.linspace(0.0, 20 * self._Ld_true / self._Rs_true, 200)
        Id_step = (Vdc_step / self._Rs_true) * (
            1.0 - np.exp(-self._Rs_true / self._Ld_true * t_dc)
        )
        # Add noise to each sample
        Id_step_noisy = Id_step * (
            1.0 + rng.normal(0.0, noise_std_frac, size=len(t_dc))
        )
        Vdc_noisy = noisy(Vdc_step)
        Id_ss_noisy = Id_step_noisy[-1]          # steady-state value
        Rs_meas = Vdc_noisy / Id_ss_noisy

        # ------ Test 2: Ld, Lq -------------------------------------------
        omega_inj = 2.0 * math.pi * _AC_INJECTION_FREQ_HZ

        # d-axis position (rotor at 0°):
        Vd_peak_true = omega_inj * self._Ld_true * 1.0   # inject Id_peak = 1 A
        Vd_peak_noisy = noisy(Vd_peak_true)
        Id_peak_noisy = noisy(1.0)
        Ld_meas = abs(Vd_peak_noisy / (omega_inj * Id_peak_noisy))

        # q-axis position (rotor at 90°):
        Vq_peak_true = omega_inj * self._Lq_true * 1.0   # inject Iq_peak = 1 A
        Vq_peak_noisy = noisy(Vq_peak_true)
        Iq_peak_noisy = noisy(1.0)
        Lq_meas = abs(Vq_peak_noisy / (omega_inj * Iq_peak_noisy))

        # AC injection waveforms (one cycle, for plotting)
        t_ac = np.linspace(0.0, 1.0 / _AC_INJECTION_FREQ_HZ, 200)
        Vd_wave = Vd_peak_noisy * np.sin(omega_inj * t_ac)
        Id_wave = Id_peak_noisy * np.sin(omega_inj * t_ac - math.pi / 2)
        Vq_wave = Vq_peak_noisy * np.sin(omega_inj * t_ac)
        Iq_wave = Iq_peak_noisy * np.sin(omega_inj * t_ac - math.pi / 2)

        # ------ Test 3: psi_f --------------------------------------------
        omega_mechs = np.array([
            rpm * math.pi / 30.0 for rpm in _BEMF_SPEEDS_RPM
        ])
        # True BEMF peak = psi_f * p * omega_mech
        bemf_true = self._psi_f_true * self._p_true * omega_mechs
        bemf_noisy = bemf_true * (
            1.0 + rng.normal(0.0, noise_std_frac, size=len(omega_mechs))
        )
        # Least-squares fit: psi_f = (B^T * B)^-1 * B^T * y
        # where B = p * omega_mechs (column vector), y = bemf_noisy
        B_vec = self._p_true * omega_mechs
        psi_f_meas = float(np.dot(B_vec, bemf_noisy) / np.dot(B_vec, B_vec))

        # ------ Assemble result ------------------------------------------
        tau_d_meas = Ld_meas / Rs_meas
        tau_q_meas = Lq_meas / Rs_meas

        results = {
            "Rs": Rs_meas,
            "Ld": Ld_meas,
            "Lq": Lq_meas,
            "psi_f": psi_f_meas,
            "tau_d": tau_d_meas,
            "tau_q": tau_q_meas,
            "source": "simulated",
            # Raw data for plotting
            "_t_dc": t_dc,
            "_Id_step": Id_step_noisy,
            "_Vdc": Vdc_noisy,
            "_t_ac": t_ac,
            "_Vd_wave": Vd_wave,
            "_Id_wave": Id_wave,
            "_Vq_wave": Vq_wave,
            "_Iq_wave": Iq_wave,
            "_omega_mechs": omega_mechs,
            "_bemf_noisy": bemf_noisy,
            "_psi_f_fit": psi_f_meas,
            "_noise_frac": noise_std_frac,
        }

        self._last_sim = results
        return results

    def simulate_mechanical(self, noise_std_frac: float = _NOISE_DEFAULT) -> dict:
        """
        Simulate four mechanical bench tests with Gaussian measurement noise.

        Requires load_cfg to have been supplied at construction.

        Test 4 — p (pole pairs):
            Spin at _POLE_PAIRS_TEST_RPM. Count BEMF zero-crossings per revolution
            to measure electrical frequency f_e = p * omega_mech / (2*pi).
            Recover p = round(f_e * 2*pi / omega_mech).  Integer rounding makes
            this robust against any realistic noise level.

        Test 5 — KE_SI (back-EMF constant, V·s/rad):
            Spin at three speeds (same as Test 3). Fit the slope of BEMF vs omega_mech:
            KE_SI = sum(BEMF_i * omega_i) / sum(omega_i^2)  (OLS, no intercept)
            Then psi_f_ke = KE_SI / p_meas.

        Test 6 — J_total (total inertia):
            No-load step: command iq = _ACCEL_TEST_IQ_A, id = 0.
            Record omega(t) = alpha_true * t + noise.  Fit slope alpha via polyfit.
            J_total = Te / alpha,  J_load = J_total - J_motor.
            Te = 1.5 * p * psi_f * iq  (exact applied current — noise enters only via omega).

        Test 7 — B_total + load coefficient (steady-state torque sweep):
            Run at _DAMPING_TEST_N_SPEEDS steady-state speed points.
            At each speed: Te_ss_true = B_total*omega + TL(omega).
            Load-specific least-squares fit (numpy lstsq):
              fan:            columns [omega, omega^2] → [B_total, k_fan]
              constant_torque: columns [omega, 1]      → [B_total, TL_const]
              position_servo:  columns [omega]         → [B_total]
            B_load = B_total - B_motor.

        Parameters
        ----------
        noise_std_frac : float
            Gaussian noise standard deviation as a fraction of the true value.

        Returns
        -------
        dict with keys: p, KE_SI, psi_f_ke, J_total, J_load, B_total, B_load,
                        k_fan (or None), TL_const (or None), source, _noise_frac,
                        plus raw measurement arrays for plotting.

        Raises
        ------
        RuntimeError
            If load_cfg was not supplied at construction.
        """
        if self.load_cfg is None:
            raise RuntimeError(
                "simulate_mechanical() requires load_cfg. "
                "Pass load_cfg when constructing ParameterIdentifier."
            )

        # Independent RNG (seed=43) — does not affect electrical test reproducibility
        rng = np.random.default_rng(seed=43)

        # ------ Test 4: pole pairs ---------------------------------------
        omega_mech_test = _POLE_PAIRS_TEST_RPM * math.pi / 30.0
        f_e_true = self._p_true * omega_mech_test / (2.0 * math.pi)
        f_e_noisy = f_e_true * (1.0 + rng.normal(0.0, noise_std_frac))
        p_meas = int(round(f_e_noisy * 2.0 * math.pi / omega_mech_test))

        # ------ Test 5: KE_SI --------------------------------------------
        omega_mechs_ke = np.array([rpm * math.pi / 30.0 for rpm in _BEMF_SPEEDS_RPM])
        bemf_true_ke = self._psi_f_true * self._p_true * omega_mechs_ke
        bemf_noisy_ke = bemf_true_ke * (
            1.0 + rng.normal(0.0, noise_std_frac, size=len(omega_mechs_ke))
        )
        # OLS: KE_SI = sum(BEMF * omega) / sum(omega^2)
        KE_SI_meas = float(
            np.dot(omega_mechs_ke, bemf_noisy_ke)
            / np.dot(omega_mechs_ke, omega_mechs_ke)
        )
        psi_f_ke = KE_SI_meas / p_meas if p_meas > 0 else float("nan")

        # ------ Test 6: J_total ------------------------------------------
        Kt_eff = 1.5 * self._p_true * self._psi_f_true
        Te_true = Kt_eff * _ACCEL_TEST_IQ_A
        alpha_true = Te_true / self._J_total_true

        t_accel = np.linspace(0.0, _ACCEL_TEST_T_S, _ACCEL_TEST_N_SAMPLES)
        omega_true_ramp = alpha_true * t_accel
        # Multiplicative noise on each speed sample (zero at t=0 is physically correct)
        omega_noisy = omega_true_ramp * (
            1.0 + rng.normal(0.0, noise_std_frac, size=_ACCEL_TEST_N_SAMPLES)
        )
        # Linear fit: omega ≈ alpha * t  (intercept free via polyfit deg=1)
        alpha_fit = float(np.polyfit(t_accel, omega_noisy, deg=1)[0])
        J_total_meas = Te_true / alpha_fit
        J_load_meas = J_total_meas - self._J_motor_true

        # ------ Test 7: B_total + load coefficients ----------------------
        omega_speeds = np.linspace(
            _DAMPING_TEST_FRAC_MIN * self._omega_rated,
            _DAMPING_TEST_FRAC_MAX * self._omega_rated,
            _DAMPING_TEST_N_SPEEDS,
        )

        # True steady-state torque at each speed
        if self._load_type == "fan":
            Te_ss_true = self._B_total_true * omega_speeds + self._k_fan_true * omega_speeds ** 2
        elif self._load_type == "constant_torque":
            Te_ss_true = self._B_total_true * omega_speeds + self._TL_const_true
        else:  # position_servo and any other
            Te_ss_true = self._B_total_true * omega_speeds

        Te_ss_noisy = Te_ss_true * (
            1.0 + rng.normal(0.0, noise_std_frac, size=_DAMPING_TEST_N_SPEEDS)
        )

        # Least-squares fit — design matrix depends on load type
        if self._load_type == "fan":
            A = np.column_stack([omega_speeds, omega_speeds ** 2])
            x, _, _, _ = np.linalg.lstsq(A, Te_ss_noisy, rcond=None)
            B_total_meas = float(x[0])
            k_fan_meas: float | None = float(x[1])
            TL_const_meas: float | None = None
        elif self._load_type == "constant_torque":
            A = np.column_stack([omega_speeds, np.ones_like(omega_speeds)])
            x, _, _, _ = np.linalg.lstsq(A, Te_ss_noisy, rcond=None)
            B_total_meas = float(x[0])
            k_fan_meas = None
            TL_const_meas = float(x[1])
        else:
            A = omega_speeds.reshape(-1, 1)
            x, _, _, _ = np.linalg.lstsq(A, Te_ss_noisy, rcond=None)
            B_total_meas = float(x[0])
            k_fan_meas = None
            TL_const_meas = None

        B_load_meas = B_total_meas - self._B_motor_true

        # ------ Assemble result ------------------------------------------
        return {
            "p":          p_meas,
            "KE_SI":      KE_SI_meas,
            "psi_f_ke":   psi_f_ke,
            "J_total":    J_total_meas,
            "J_load":     J_load_meas,
            "B_total":    B_total_meas,
            "B_load":     B_load_meas,
            "k_fan":      k_fan_meas,
            "TL_const":   TL_const_meas,
            "source":     "simulated_mechanical",
            "_noise_frac":       noise_std_frac,
            # Raw data for plot_mechanical()
            "_f_e_noisy":        f_e_noisy,
            "_omega_mech_test":  omega_mech_test,
            "_omega_mechs_ke":   omega_mechs_ke,
            "_bemf_noisy_ke":    bemf_noisy_ke,
            "_KE_SI_fit":        KE_SI_meas,
            "_t_accel":          t_accel,
            "_omega_accel":      omega_noisy,
            "_alpha_fit":        alpha_fit,
            "_omega_sweep":      omega_speeds,
            "_Te_ss_noisy":      Te_ss_noisy,
            "_B_total_fit":      B_total_meas,
        }

    def override(
        self,
        Rs: float = None,
        Ld: float = None,
        Lq: float = None,
        psi_f: float = None,
        # Mechanical parameters (require load_cfg at construction)
        p: int = None,
        KE_SI: float = None,
        J_total: float = None,
        J_load: float = None,
        B_total: float = None,
        B_load: float = None,
        k_fan: float = None,
        TL_const: float = None,
    ) -> dict:
        """
        Accept directly measured parameters and validate physical consistency.

        Electrical validation checks:
          1. All electrical values must be > 0.
          2. tau_e = Ld/Rs must be within ±25% of ground-truth tau_e.
          3. Lq >= Ld (SPMSM: Lq ≈ Ld; IPMSM: Lq > Ld).

        Mechanical validation checks (requires load_cfg):
          4. p must be an integer >= 1.
          5. KE_SI > 0.
          6. J_total > 0 and >= J_motor_true (cannot be less than motor alone).
          7. J_load >= 0; if both J_total and J_load given, must be consistent.
          8. B_total >= 0 and >= B_motor_true.
          9. B_load >= 0.
          10. k_fan >= 0 if provided.
          11. TL_const >= 0 if provided.

        Parameters
        ----------
        Rs, Ld, Lq, psi_f : float, optional
            Electrical measurements. Unspecified values fall back to ground truth.
        p, KE_SI, J_total, J_load, B_total, B_load, k_fan, TL_const : optional
            Mechanical measurements. Require load_cfg. Unspecified → ground truth.

        Returns
        -------
        dict with all identified keys and source='override'.

        Raises
        ------
        ValueError
            If any consistency check fails.
        RuntimeError
            If mechanical parameters are supplied but load_cfg was not given.
        """
        # --- Electrical fallbacks and validation ---
        Rs    = Rs    if Rs    is not None else self._Rs_true
        Ld    = Ld    if Ld    is not None else self._Ld_true
        Lq    = Lq    if Lq    is not None else self._Lq_true
        psi_f = psi_f if psi_f is not None else self._psi_f_true

        for name, val in [("Rs", Rs), ("Ld", Ld), ("Lq", Lq), ("psi_f", psi_f)]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

        tau_e_meas = Ld / Rs
        tau_e_ref  = self._Ld_true / self._Rs_true
        rel_err    = abs(tau_e_meas - tau_e_ref) / tau_e_ref
        if rel_err > _TAU_E_TOLERANCE:
            raise ValueError(
                f"tau_e = Ld/Rs = {tau_e_meas*1e3:.2f} ms differs from "
                f"expected {tau_e_ref*1e3:.2f} ms by {rel_err*100:.1f}% "
                f"(tolerance {_TAU_E_TOLERANCE*100:.0f}%)"
            )

        if Lq < Ld * (1.0 - 0.02):
            raise ValueError(
                f"Lq ({Lq*1e3:.3f} mH) < Ld ({Ld*1e3:.3f} mH). "
                f"PMSM requires Lq >= Ld."
            )

        result = {
            "Rs": Rs, "Ld": Ld, "Lq": Lq, "psi_f": psi_f,
            "tau_d": Ld / Rs, "tau_q": Lq / Rs,
            "source": "override",
        }

        # --- Mechanical parameters ---
        _mech_supplied = any(v is not None for v in
                             [p, KE_SI, J_total, J_load, B_total, B_load, k_fan, TL_const])
        if _mech_supplied and self.load_cfg is None:
            raise RuntimeError(
                "Mechanical parameter override requires load_cfg. "
                "Pass load_cfg when constructing ParameterIdentifier."
            )

        if self.load_cfg is not None:
            # Fallbacks
            p       = p       if p       is not None else self._p_true
            KE_SI   = KE_SI   if KE_SI   is not None else self._psi_f_true * self._p_true
            J_total = J_total if J_total is not None else self._J_total_true
            J_load  = J_load  if J_load  is not None else self._J_load_true
            B_total = B_total if B_total is not None else self._B_total_true
            B_load  = B_load  if B_load  is not None else self._B_load_true
            k_fan   = k_fan   if k_fan   is not None else self._k_fan_true
            TL_const = TL_const if TL_const is not None else self._TL_const_true

            # Validations
            if not (isinstance(p, int) and p >= 1):
                raise ValueError(f"p must be an integer >= 1, got {p!r}")
            if KE_SI <= 0:
                raise ValueError(f"KE_SI must be > 0, got {KE_SI}")
            if J_total <= 0:
                raise ValueError(f"J_total must be > 0, got {J_total}")
            if J_total < self._J_motor_true:
                raise ValueError(
                    f"J_total ({J_total:.4e}) < J_motor ({self._J_motor_true:.4e}). "
                    f"Total inertia cannot be less than motor inertia alone."
                )
            if J_load < 0:
                raise ValueError(f"J_load must be >= 0, got {J_load}")
            # Cross-check J_total vs J_load consistency
            implied_J_load = J_total - self._J_motor_true
            if abs(J_load - implied_J_load) / max(self._J_motor_true, 1e-12) > _J_CONSISTENCY_TOL:
                raise ValueError(
                    f"J_load ({J_load:.4e}) inconsistent with "
                    f"J_total - J_motor = {implied_J_load:.4e} "
                    f"(tolerance {_J_CONSISTENCY_TOL*100:.0f}%)"
                )
            if B_total < 0:
                raise ValueError(f"B_total must be >= 0, got {B_total}")
            if B_total < self._B_motor_true:
                raise ValueError(
                    f"B_total ({B_total:.4e}) < B_motor ({self._B_motor_true:.4e}). "
                    f"Total damping cannot be less than motor damping alone."
                )
            if B_load < 0:
                raise ValueError(f"B_load must be >= 0, got {B_load}")
            if k_fan is not None and k_fan < 0:
                raise ValueError(f"k_fan must be >= 0, got {k_fan}")
            if TL_const is not None and TL_const < 0:
                raise ValueError(f"TL_const must be >= 0, got {TL_const}")

            result.update({
                "p":        p,
                "KE_SI":    KE_SI,
                "psi_f_ke": KE_SI / p,
                "J_total":  J_total,
                "J_load":   J_load,
                "B_total":  B_total,
                "B_load":   B_load,
                "k_fan":    k_fan,
                "TL_const": TL_const,
            })

        return result

    # ------------------------------------------------------------------
    # Plotting — electrical
    # ------------------------------------------------------------------

    def plot_identification(self, results: dict, save_path: str = None) -> None:
        """
        Plot the three simulated electrical bench tests.

        Layout: 1 row × 3 columns
          [0] DC step response (Rs test)
          [1] AC injection waveforms (Ld/Lq test)
          [2] BEMF vs speed fit (psi_f test)

        Parameters
        ----------
        results : dict
            Output from simulate(). No-op if source == 'override'.
        save_path : str, optional
            If given, saves to this path. Otherwise shows interactively.
        """
        if results.get("source") == "override":
            print("plot_identification: no raw data for override results — skipping.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(
            f"Parameter Identification — {self.motor_cfg['name']} "
            f"[noise={results['_noise_frac']*100:.0f}%]",
            fontsize=12,
        )

        # --- Plot 1: DC step response (Rs) ---
        ax = axes[0]
        t_ms = results["_t_dc"] * 1e3
        Id = results["_Id_step"]
        Vdc = results["_Vdc"]
        Id_ss = Id[-1]
        Rs_meas = Vdc / Id_ss
        ax.plot(t_ms, Id, color="steelblue", linewidth=1.5, label="Id (noisy)")
        ax.axhline(Id_ss, color="crimson", linestyle="--", linewidth=1,
                   label=f"Id_ss = {Id_ss:.3f} A")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Id (A)")
        ax.set_title(
            f"Test 1 — Rs\nVdc={Vdc:.3f} V  →  Rs={Rs_meas*1000:.2f} m\u03a9"
            f"\n(true={self._Rs_true*1000:.2f} m\u03a9)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Plot 2: AC injection waveforms (Ld / Lq) ---
        ax = axes[1]
        t_us = results["_t_ac"] * 1e6
        ax.plot(t_us, results["_Vd_wave"], color="steelblue", linewidth=1.5,
                label="Vd (d-axis)")
        ax.plot(t_us, results["_Id_wave"] * max(abs(results["_Vd_wave"])),
                color="steelblue", linewidth=1, linestyle="--", label="Id (scaled)")
        ax.plot(t_us, results["_Vq_wave"], color="darkorange", linewidth=1.5,
                label="Vq (q-axis)")
        ax.plot(t_us, results["_Iq_wave"] * max(abs(results["_Vq_wave"])),
                color="darkorange", linewidth=1, linestyle="--", label="Iq (scaled)")
        ax.set_xlabel("Time (\u03bcs)")
        ax.set_ylabel("Amplitude")
        Ld_meas = results["Ld"]
        Lq_meas = results["Lq"]
        ax.set_title(
            f"Test 2 — Ld, Lq  @{_AC_INJECTION_FREQ_HZ:.0f} Hz\n"
            f"Ld={Ld_meas*1e3:.3f} mH (true={self._Ld_true*1e3:.3f})\n"
            f"Lq={Lq_meas*1e3:.3f} mH (true={self._Lq_true*1e3:.3f})"
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # --- Plot 3: BEMF fit (psi_f) ---
        ax = axes[2]
        omega_mechs = results["_omega_mechs"]
        bemf_noisy = results["_bemf_noisy"]
        psi_f_fit = results["_psi_f_fit"]

        omega_plot = np.linspace(0, max(omega_mechs) * 1.1, 100)
        bemf_fit_line = psi_f_fit * self._p_true * omega_plot

        ax.scatter(omega_mechs, bemf_noisy, color="steelblue", zorder=5,
                   label="Measured BEMF")
        ax.plot(omega_plot, bemf_fit_line, color="crimson", linewidth=1.5,
                label=f"Fit: psi_f={psi_f_fit*1e3:.2f} mWb")
        ax.plot(omega_plot, self._psi_f_true * self._p_true * omega_plot,
                color="gray", linewidth=1, linestyle="--",
                label=f"True: psi_f={self._psi_f_true*1e3:.2f} mWb")
        ax.set_xlabel("omega_mech (rad/s)")
        ax.set_ylabel("BEMF peak (V)")
        ax.set_title(
            f"Test 3 — psi_f (BEMF fit)\n"
            f"psi_f={psi_f_fit*1e3:.3f} mWb "
            f"(true={self._psi_f_true*1e3:.3f} mWb)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    # ------------------------------------------------------------------
    # Plotting — mechanical
    # ------------------------------------------------------------------

    def plot_mechanical(self, results_mech: dict, save_path: str = None) -> None:
        """
        Plot the four simulated mechanical bench tests.

        Layout: 2 rows × 2 columns
          [0,0] Test 4 — Pole pairs (electrical frequency annotation)
          [0,1] Test 5 — KE_SI (BEMF scatter + fit line)
          [1,0] Test 6 — J_total (omega ramp + fitted slope)
          [1,1] Test 7 — B_total / load coefficients (Te_ss scatter + fit)

        Parameters
        ----------
        results_mech : dict
            Output from simulate_mechanical().
        save_path : str, optional
            If given, saves figure to this path.
        """
        if results_mech.get("source") != "simulated_mechanical":
            print("plot_mechanical: expected source='simulated_mechanical' — skipping.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(
            f"Mechanical Identification — {self.motor_cfg['name']} "
            f"[noise={results_mech['_noise_frac']*100:.0f}%]",
            fontsize=12,
        )

        # --- [0,0] Test 4: Pole pairs ---
        ax = axes[0, 0]
        f_e_noisy = results_mech["_f_e_noisy"]
        omega_t   = results_mech["_omega_mech_test"]
        f_e_true  = self._p_true * omega_t / (2.0 * math.pi)
        p_meas    = results_mech["p"]
        bars = ax.bar(["f_e true", "f_e measured"],
                      [f_e_true, f_e_noisy],
                      color=["gray", "steelblue"], width=0.4)
        ax.bar_label(bars, fmt="%.2f Hz", padding=3, fontsize=9)
        ax.set_ylabel("Electrical frequency (Hz)")
        ax.set_title(
            f"Test 4 — Pole pairs\n"
            f"f_e_true={f_e_true:.2f} Hz  →  p = {p_meas}  (true={self._p_true})"
        )
        ax.grid(True, axis="y", alpha=0.3)

        # --- [0,1] Test 5: KE_SI ---
        ax = axes[0, 1]
        omega_ke  = results_mech["_omega_mechs_ke"]
        bemf_ke   = results_mech["_bemf_noisy_ke"]
        KE_SI_fit = results_mech["_KE_SI_fit"]
        KE_SI_true = self._psi_f_true * self._p_true

        omega_plot = np.linspace(0, max(omega_ke) * 1.1, 100)
        ax.scatter(omega_ke, bemf_ke, color="steelblue", zorder=5, label="Measured BEMF")
        ax.plot(omega_plot, KE_SI_fit * omega_plot, color="crimson", linewidth=1.5,
                label=f"Fit: KE_SI={KE_SI_fit:.4f} V·s/rad")
        ax.plot(omega_plot, KE_SI_true * omega_plot, color="gray", linewidth=1,
                linestyle="--", label=f"True: KE_SI={KE_SI_true:.4f} V·s/rad")
        ax.set_xlabel("omega_mech (rad/s)")
        ax.set_ylabel("BEMF peak (V)")
        ax.set_title(
            f"Test 5 — KE_SI\n"
            f"KE_SI={KE_SI_fit:.4f} V·s/rad  "
            f"psi_f_ke={results_mech['psi_f_ke']*1e3:.2f} mWb"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- [1,0] Test 6: J_total ---
        ax = axes[1, 0]
        t_accel = results_mech["_t_accel"]
        omega_a = results_mech["_omega_accel"]
        alpha_f = results_mech["_alpha_fit"]

        ax.scatter(t_accel[::10], omega_a[::10], color="steelblue", s=8,
                   zorder=5, label="omega (noisy, 1-in-10)")
        ax.plot(t_accel, alpha_f * t_accel, color="crimson", linewidth=1.5,
                label=f"Fit: α={alpha_f:.3f} rad/s²")
        Kt_eff = 1.5 * self._p_true * self._psi_f_true
        alpha_true = Kt_eff * _ACCEL_TEST_IQ_A / self._J_total_true
        ax.plot(t_accel, alpha_true * t_accel, color="gray", linewidth=1,
                linestyle="--", label=f"True: α={alpha_true:.3f} rad/s²")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("omega_mech (rad/s)")
        ax.set_title(
            f"Test 6 — J_total\n"
            f"J_total={results_mech['J_total']:.4e} kg·m²  "
            f"(true={self._J_total_true:.4e})"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- [1,1] Test 7: B_total + load coefficients ---
        ax = axes[1, 1]
        omega_sw = results_mech["_omega_sweep"]
        Te_sw    = results_mech["_Te_ss_noisy"]
        B_fit    = results_mech["_B_total_fit"]
        k_fan_f  = results_mech.get("k_fan")
        TL_f     = results_mech.get("TL_const")

        omega_plot = np.linspace(omega_sw[0] * 0.9, omega_sw[-1] * 1.05, 200)
        if k_fan_f is not None:
            Te_fit_line = B_fit * omega_plot + k_fan_f * omega_plot ** 2
            Te_true_line = self._B_total_true * omega_plot + self._k_fan_true * omega_plot ** 2
            load_label = f"fan: B={B_fit:.4f}, k={k_fan_f:.2e}"
        elif TL_f is not None:
            Te_fit_line = B_fit * omega_plot + TL_f
            Te_true_line = self._B_total_true * omega_plot + self._TL_const_true
            load_label = f"const-torque: B={B_fit:.4f}, TL={TL_f:.3f}"
        else:
            Te_fit_line = B_fit * omega_plot
            Te_true_line = self._B_total_true * omega_plot
            load_label = f"servo: B={B_fit:.4f}"

        ax.scatter(omega_sw, Te_sw, color="steelblue", zorder=5, label="Te_ss (noisy)")
        ax.plot(omega_plot, Te_fit_line, color="crimson", linewidth=1.5,
                label=f"Fit ({load_label})")
        ax.plot(omega_plot, Te_true_line, color="gray", linewidth=1, linestyle="--",
                label=f"True: B={self._B_total_true:.4f}")
        ax.set_xlabel("omega_mech (rad/s)")
        ax.set_ylabel("Steady-state torque (N·m)")
        ax.set_title(
            f"Test 7 — B_total\n"
            f"B_total={B_fit:.5f} N·m·s/rad  "
            f"(true={self._B_total_true:.5f})"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close(fig)

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------

    def print_comparison(self, results: dict) -> None:
        """
        Print a table comparing measured vs true parameters.

        Prints the electrical section always.  If mechanical keys (J_total, etc.)
        are present in results, also prints a mechanical section.

        Parameters
        ----------
        results : dict
            Output from simulate(), simulate_mechanical(), override(),
            or a merged dict: {**elec_results, **mech_results}.
        """
        # --- Electrical section (only when electrical keys are present) ---
        has_elec = "Rs" in results
        if not has_elec and "J_total" not in results:
            print("print_comparison: unrecognised results dict — no electrical or mechanical keys.")
            return

        if has_elec:
            true_e = {
                "Rs": self._Rs_true, "Ld": self._Ld_true,
                "Lq": self._Lq_true, "psi_f": self._psi_f_true,
            }
            meas_e = {k: results[k] for k in ("Rs", "Ld", "Lq", "psi_f")}
            units_e = {"Rs": "Ohm", "Ld": "mH", "Lq": "mH", "psi_f": "mWb"}
            scale_e = {"Rs": 1.0, "Ld": 1e3, "Lq": 1e3, "psi_f": 1e3}

            print(f"\nParameter Identification Summary [{results.get('source', '?')}]")
            print(f"Motor: {self.motor_cfg['name']}")
            print(f"{'Param':<10} {'True':>12} {'Measured':>12} {'Error %':>8}  Unit")
            print("-" * 52)
            for key in ("Rs", "Ld", "Lq", "psi_f"):
                t = true_e[key] * scale_e[key]
                m = meas_e[key] * scale_e[key]
                err = (m - t) / t * 100.0
                print(f"{key:<10} {t:>12.4f} {m:>12.4f} {err:>+8.2f}%  {units_e[key]}")
            print(f"\ntau_d = {results['tau_d']*1e3:.3f} ms")
            print(f"tau_q = {results['tau_q']*1e3:.3f} ms")

        # --- Mechanical section (only if mechanical keys present) ---
        if "J_total" not in results:
            return

        if self.load_cfg is None:
            # No ground-truth available — print measured values only
            print(f"\nMechanical Identification (no ground truth — load_cfg not supplied)")
            print(f"  p       = {results['p']}")
            print(f"  KE_SI   = {results['KE_SI']:.4f} V.s/rad")
            print(f"  J_total = {results['J_total']:.4e} kg.m2")
            print(f"  B_total = {results['B_total']:.5f} N.m.s/rad")
            return

        print(f"\nMechanical Identification Summary")
        print(f"  Motor:  J_motor={self._J_motor_true:.4e} kg.m2  "
              f"B_motor={self._B_motor_true:.5f} N.m.s/rad")
        print(f"  Load:   J_load_true={self._J_load_true:.4e} kg.m2  "
              f"B_load_true={self._B_load_true:.5f} N.m.s/rad")

        KE_SI_true = self._psi_f_true * self._p_true

        rows = [
            # (label, true_val, meas_val, scale, unit)
            ("p",        self._p_true,     results["p"],       1.0,  "-"),
            ("KE_SI",    KE_SI_true,       results["KE_SI"],   1.0,  "V.s/rad"),
            ("psi_f_ke", self._psi_f_true, results["psi_f_ke"], 1e3, "mWb"),
            ("J_total",  self._J_total_true, results["J_total"], 1e4, "x1e-4 kg.m2"),
            ("J_load",   self._J_load_true,  results["J_load"],  1e4, "x1e-4 kg.m2"),
            ("B_total",  self._B_total_true, results["B_total"], 1e3, "mN.m.s/rad"),
            ("B_load",   self._B_load_true,  results["B_load"],  1e3, "mN.m.s/rad"),
        ]
        if results.get("k_fan") is not None:
            rows.append(("k_fan", self._k_fan_true, results["k_fan"], 1e5,
                          "x1e-5 N.m/(r/s)^2"))
        if results.get("TL_const") is not None:
            rows.append(("TL_const", self._TL_const_true, results["TL_const"], 1.0,
                          "N.m"))

        print(f"\n{'Param':<10} {'True':>14} {'Measured':>14} {'Error %':>8}  Unit")
        print("-" * 58)
        for label, t_raw, m_raw, sc, unit in rows:
            t = t_raw * sc
            m = m_raw * sc
            if abs(t) > 1e-15:
                err_str = f"{(m - t) / t * 100.0:>+8.2f}%"
            else:
                err_str = "     N/A "
            print(f"{label:<10} {t:>14.4f} {m:>14.4f} {err_str}  {unit}")
