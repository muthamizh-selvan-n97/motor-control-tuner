"""
modules/param_id.py — Simulated parameter identification for PMSM.

Two-path design:
  1. simulate() — inject test signals with Gaussian noise, extract Rs/Ld/Lq/psi_f
  2. override() — accept measured values, validate physical consistency

Both paths return the same dict:
  {'Rs': float, 'Ld': float, 'Lq': float, 'psi_f': float,
   'tau_d': float, 'tau_q': float, 'source': str}

Bench tests simulated:
  Test 1 — Rs : DC lockout step, measure steady-state Id
  Test 2 — Ld, Lq : AC standstill injection at 100 Hz, two rotor positions
  Test 3 — psi_f : BEMF open-circuit measurement at three speeds
"""

import math

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_AC_INJECTION_FREQ_HZ = 100.0       # standstill injection frequency
_BEMF_SPEEDS_RPM = (1000.0, 2000.0, 3000.0)   # open-circuit BEMF test speeds
_TAU_E_TOLERANCE = 0.25             # ±25% tolerance on tau_e consistency check
_NOISE_DEFAULT = 0.02               # 2% default noise std fraction

# Consistency check thresholds
_MIN_SALIENCY = 1.0                 # Lq/Ld must be >= 1.0 (Lq >= Ld for IPMSM, = for SPMSM)


class ParameterIdentifier:
    """
    Simulate or accept measured PMSM electrical parameters.

    Parameters
    ----------
    motor_cfg : dict
        Validated motor config from utils.config.load_config().
        Used as the ground-truth plant for simulated tests.
    """

    def __init__(self, motor_cfg: dict) -> None:
        self.motor_cfg = motor_cfg
        self.motor_type: str = motor_cfg["motor_type"]

        # Ground-truth parameters (used only by simulate())
        elec = motor_cfg["electrical"]
        self._Rs_true: float = elec["Rs_ohm"]
        self._Ld_true: float = elec["Ld_H"]
        self._Lq_true: float = elec["Lq_H"]
        self._psi_f_true: float = elec["psi_f_Wb"]
        self._p_true: int = elec["pole_pairs"]

        # Storage for last simulation run (for plotting)
        self._last_sim: dict = {}

    # ------------------------------------------------------------------
    # Public API
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

    def override(
        self,
        Rs: float = None,
        Ld: float = None,
        Lq: float = None,
        psi_f: float = None,
    ) -> dict:
        """
        Accept directly measured parameters and validate physical consistency.

        Validation checks:
          1. All values must be > 0.
          2. tau_e = Ld/Rs must be within ±25% of ground-truth tau_e.
          3. Lq >= Ld (SPMSM: Lq ≈ Ld; IPMSM: Lq > Ld).
          4. psi_f > 0.

        Parameters
        ----------
        Rs : float   Measured phase resistance in Ohm.
        Ld : float   Measured d-axis inductance in H.
        Lq : float   Measured q-axis inductance in H.
        psi_f : float  Measured flux linkage in Wb.

        Returns
        -------
        dict with keys: Rs, Ld, Lq, psi_f, tau_d, tau_q, source.

        Raises
        ------
        ValueError
            If any consistency check fails.
        """
        # Fall back to ground-truth if a value is not provided
        Rs = Rs if Rs is not None else self._Rs_true
        Ld = Ld if Ld is not None else self._Ld_true
        Lq = Lq if Lq is not None else self._Lq_true
        psi_f = psi_f if psi_f is not None else self._psi_f_true

        # Check 1: positivity
        for name, val in [("Rs", Rs), ("Ld", Ld), ("Lq", Lq), ("psi_f", psi_f)]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

        # Check 2: tau_e consistency with ground truth
        tau_e_meas = Ld / Rs
        tau_e_ref = self._Ld_true / self._Rs_true
        rel_err = abs(tau_e_meas - tau_e_ref) / tau_e_ref
        if rel_err > _TAU_E_TOLERANCE:
            raise ValueError(
                f"tau_e = Ld/Rs = {tau_e_meas*1e3:.2f} ms differs from "
                f"expected {tau_e_ref*1e3:.2f} ms by {rel_err*100:.1f}% "
                f"(tolerance {_TAU_E_TOLERANCE*100:.0f}%)"
            )

        # Check 3: saliency (Lq >= Ld)
        if Lq < Ld * (1.0 - 0.02):   # allow 2% float tolerance
            raise ValueError(
                f"Lq ({Lq*1e3:.3f} mH) < Ld ({Ld*1e3:.3f} mH). "
                f"PMSM requires Lq >= Ld."
            )

        # Check 4: psi_f (already covered by positivity check)

        tau_d = Ld / Rs
        tau_q = Lq / Rs

        return {
            "Rs": Rs,
            "Ld": Ld,
            "Lq": Lq,
            "psi_f": psi_f,
            "tau_d": tau_d,
            "tau_q": tau_q,
            "source": "override",
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_identification(self, results: dict, save_path: str = None) -> None:
        """
        Plot the three simulated bench tests.

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

        # Fitted line
        omega_plot = np.linspace(0, max(omega_mechs) * 1.1, 100)
        bemf_fit_line = psi_f_fit * self._p_true * omega_plot

        ax.scatter(omega_mechs, bemf_noisy, color="steelblue", zorder=5,
                   label="Measured BEMF")
        ax.plot(omega_plot, bemf_fit_line, color="crimson", linewidth=1.5,
                label=f"Fit: psi_f={psi_f_fit*1e3:.2f} mWb")
        # True line for reference
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
    # Comparison summary
    # ------------------------------------------------------------------

    def print_comparison(self, results: dict) -> None:
        """
        Print a table comparing measured vs true parameters.

        Parameters
        ----------
        results : dict
            Output from simulate() or override().
        """
        true = {
            "Rs": self._Rs_true,
            "Ld": self._Ld_true,
            "Lq": self._Lq_true,
            "psi_f": self._psi_f_true,
        }
        meas = {k: results[k] for k in ("Rs", "Ld", "Lq", "psi_f")}
        units = {"Rs": "Ohm", "Ld": "mH", "Lq": "mH", "psi_f": "mWb"}
        scale = {"Rs": 1.0, "Ld": 1e3, "Lq": 1e3, "psi_f": 1e3}

        print(f"\nParameter Identification Summary [{results['source']}]")
        print(f"Motor: {self.motor_cfg['name']}")
        print(f"{'Param':<8} {'True':>10} {'Measured':>10} {'Error %':>8}  Unit")
        print("-" * 48)
        for key in ("Rs", "Ld", "Lq", "psi_f"):
            t = true[key] * scale[key]
            m = meas[key] * scale[key]
            err = (m - t) / t * 100.0
            print(f"{key:<8} {t:>10.4f} {m:>10.4f} {err:>+8.2f}%  {units[key]}")
        print(f"\ntau_d = {results['tau_d']*1e3:.3f} ms")
        print(f"tau_q = {results['tau_q']*1e3:.3f} ms")
