"""
modules/speed_loop.py — PI speed loop tuner for PMSM.

Four tuning methods (same interface as CurrentLoopTuner):
  1. pole_zero        — analytical (type-II plant → modified cancellation)
  2. frequency_domain — place gain crossover at desired BW
  3. root_locus       — place closed-loop poles at desired damping ratio
  4. ziegler_nichols  — find Ku/Tu with current-loop lag included

Speed plant (with closed current loop as first-order lag):
    G_w(s) = (Kt_eff / J_total) / (s * (tau_cl*s + 1))

where:
    Kt_eff  = (3/2) * p * psi_f      (effective torque constant)
    tau_cl  = 1 / (2*pi*BW_current)  (current loop closed-loop time constant)
    J_total = J_motor + J_load
    BW_hierarchy: BW_speed <= BW_current / 10

Fan load linearisation:
    T_L(omega) = k_fan * omega^2
    Linearised at rated speed: TL_lin = 2 * k_fan * omega_rated * delta_omega
    Adds effective damping: B_eff = B_total + 2*k_fan*omega_rated

Anti-windup (back-calculation):
    Tt = Ti  (where Ti = Kp/Ki is the integral time constant)
    Back-calculation gain: Kb = 1/Tt = Ki/Kp
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from modules.plant import PMSMPlant
from modules.current_loop import LoopResult

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_PM_MIN_DEG: float = 45.0       # minimum acceptable phase margin
_GM_MIN_DB: float = 6.0         # minimum acceptable gain margin
_BW_RATIO: float = 10.0         # speed BW must be <= current loop BW / 10
_SETTLE_BAND: float = 0.02      # 2% settling band
_DEFAULT_TAU_CL_S: float = 10e-3  # default speed loop time constant (10 ms)
_DEFAULT_BW_HZ: float = 50.0    # default crossover BW for freq-domain method
_DEFAULT_ZETA: float = 0.707    # default damping ratio for root-locus method
_DEFAULT_TS_S: float = 50e-6    # default PWM period for Z-N method


# ---------------------------------------------------------------------------
# SpeedLoopResult dataclass (extends LoopResult with speed-loop extras)
# ---------------------------------------------------------------------------

@dataclass
class SpeedLoopResult(LoopResult):
    """
    Returned by SpeedLoopTuner.tune().
    Extends LoopResult with anti-windup parameter and plant data.
    """
    Kb: float = 0.0          # Anti-windup back-calculation gain (= Ki/Kp)
    Kt_eff: float = 0.0      # Effective torque constant (3/2)*p*psi_f
    J_total: float = 0.0     # Total inertia (motor + load)
    B_eff: float = 0.0       # Effective damping (incl. fan linearisation)
    tau_cl: float = 0.0      # Current loop closed-loop time constant (s)

    def summary(self) -> None:
        """Print summary table to stdout."""
        hdr = (
            f"{'Method':<18} {'Axis':<6} {'Kp':>8} {'Ki':>8} "
            f"{'BW_Hz':>7} {'PM_deg':>7} {'GM_dB':>7} {'Tset_ms':>8} {'Kb':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        gm_str = f"{self.GM_dB:>7.1f}" if self.GM_dB < 900 else f"{'inf':>7}"
        print(
            f"{self.method:<18} {self.axis:<6} {self.Kp:>8.4f} {self.Ki:>8.4f} "
            f"{self.BW_Hz:>7.1f} {self.PM_deg:>7.1f} {gm_str} "
            f"{self.settling_ms:>8.3f} {self.Kb:>8.4f}"
        )
        for w in self.warnings:
            print(f"  {w}")


# ---------------------------------------------------------------------------
# SpeedLoopTuner
# ---------------------------------------------------------------------------

class SpeedLoopTuner:
    """
    Tune the PI speed controller using four methods.

    The speed plant includes the closed current loop as a first-order lag.
    Fan loads are linearised around rated speed to give an effective damping term.

    Parameters
    ----------
    plant : PMSMPlant
        PMSM plant model.
    params : dict
        Identified parameters from ParameterIdentifier (Rs, Ld, Lq, psi_f).
        If None, falls back to plant parameters.
    i_loop : LoopResult
        Tuned current loop result (used for BW hierarchy check and tau_cl).
    """

    def __init__(
        self,
        plant: PMSMPlant,
        params: dict,
        i_loop: LoopResult,
    ) -> None:
        self.plant = plant
        self.i_loop = i_loop

        # Electrical / mechanical parameters
        if params is not None:
            psi_f = float(params.get("psi_f", plant.psi_f))
        else:
            psi_f = plant.psi_f

        p = plant.p
        self._Kt_eff: float = 1.5 * p * psi_f        # (3/2)*p*psi_f

        mech = plant.motor_cfg["mechanical"]
        load = plant.load_cfg
        self._J_total: float = mech["J_kgm2"] + load.get("J_load_kgm2", 0.0)
        self._B_total: float = (
            mech.get("B_Nms_rad", 0.0) + load.get("B_load_Nms_rad", 0.0)
        )

        # Fan load linearisation: add 2*k_fan*omega_rated as effective damping
        self._B_eff: float = self._B_total
        if load.get("load_type") == "fan":
            k_fan = float(load.get("k_fan", 0.0))
            omega_rated = float(
                plant.motor_cfg["rated"].get("speed_rad_s",
                    plant.motor_cfg["rated"]["speed_rpm"] * math.pi / 30.0)
            )
            self._B_eff += 2.0 * k_fan * omega_rated

        # Current loop closed-loop time constant
        if i_loop.BW_Hz > 0:
            self._tau_cl: float = 1.0 / (2.0 * math.pi * i_loop.BW_Hz)
        else:
            # Fallback: tau_cl from settling time
            self._tau_cl = max(i_loop.settling_ms * 1e-3 / 5.0, 1e-4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(
        self,
        method: str = "pole_zero",
        **kwargs,
    ) -> SpeedLoopResult:
        """
        Tune the speed loop PI controller.

        Parameters
        ----------
        method : str
            'pole_zero', 'frequency_domain', 'root_locus', or 'ziegler_nichols'.
        **kwargs
            tau_cl_s (pole_zero, root_locus), BW_Hz (frequency_domain),
            target_zeta (root_locus), Ts_s (ziegler_nichols).

        Returns
        -------
        SpeedLoopResult
            Gains plus full verification suite.
        """
        if method == "pole_zero":
            tau_w_s = kwargs.get("tau_cl_s", _DEFAULT_TAU_CL_S)
            Kp, Ki = self._tune_pole_zero(tau_w_s)

        elif method == "frequency_domain":
            BW_Hz = kwargs.get("BW_Hz", _DEFAULT_BW_HZ)
            Kp, Ki = self._tune_frequency_domain(BW_Hz)

        elif method == "root_locus":
            target_zeta = kwargs.get("target_zeta", _DEFAULT_ZETA)
            tau_w_s = kwargs.get("tau_cl_s", _DEFAULT_TAU_CL_S)
            Kp, Ki = self._tune_root_locus(target_zeta, tau_w_s)

        elif method == "ziegler_nichols":
            Ts_s = kwargs.get("Ts_s", _DEFAULT_TS_S)
            Kp, Ki = self._tune_ziegler_nichols(Ts_s)

        else:
            raise ValueError(
                f"method must be one of 'pole_zero', 'frequency_domain', "
                f"'root_locus', 'ziegler_nichols'. Got '{method}'."
            )

        return self._verify(Kp, Ki, method)

    # ------------------------------------------------------------------
    # Tuning methods
    # ------------------------------------------------------------------

    def _tune_pole_zero(self, tau_w_s: float) -> tuple[float, float]:
        """
        Speed loop pole-zero cancellation.

        Speed plant (with B_eff > 0 visible as stable pole):
            G_w(s) = (Kt_eff/J_total) / (s * (tau_cl*s + 1))

        For a pure integrator plant (B_eff ≈ 0), the PI controller adds one
        integrator → type-II system. Bandwidth-based design:
            Kp = J_total / (Kt_eff * tau_w_s)
            Ki = Kp * B_eff / J_total   if B_eff > 0
               = Kp / (10 * tau_w_s)    otherwise (decade below crossover)

        This ensures the integral time is long relative to the mechanical time
        constant while placing the gain crossover at approximately 1/tau_w_s.
        """
        Kp = self._J_total / (self._Kt_eff * tau_w_s)
        if self._B_eff > 1e-9:
            Ki = Kp * self._B_eff / self._J_total
        else:
            Ki = Kp / (10.0 * tau_w_s)
        return Kp, Ki

    def _tune_frequency_domain(self, BW_Hz: float) -> tuple[float, float]:
        """
        Place gain crossover at desired bandwidth.

        At omega_c = 2*pi*BW_Hz, |C(j*omega_c) * G_w(j*omega_c)| = 1.

        Speed plant magnitude at omega_c (with tau_cl lag):
            |G_w(j*omega_c)| = (Kt_eff/J_total) / (omega_c * sqrt(1 + (tau_cl*omega_c)^2))

        With Ki = Kp * omega_c / 10:
            |C(j*omega_c)| ≈ Kp * sqrt(1.01)

        Solving for Kp:
            Kp = J_total * omega_c * sqrt(1 + (tau_cl*omega_c)^2)
                 / (Kt_eff * sqrt(1.01))
        Ki = Kp * omega_c / 10
        """
        omega_c = 2.0 * math.pi * BW_Hz
        plant_mag = (self._Kt_eff / self._J_total) / (
            omega_c * math.sqrt(1.0 + (self._tau_cl * omega_c) ** 2)
        )
        Kp = 1.0 / (plant_mag * math.sqrt(1.01))
        Ki = Kp * omega_c / 10.0
        return Kp, Ki

    def _tune_root_locus(
        self, target_zeta: float, tau_w_s: float
    ) -> tuple[float, float]:
        """
        Place closed-loop poles at desired damping ratio.

        Simplified speed plant (ignoring tau_cl for pole placement):
            G_w(s) = (Kt_eff/J_total) / s

        PI + integrator plant gives characteristic equation:
            s^2 + (Kp*Kt_eff/J_total)*s + (Ki*Kt_eff/J_total) = 0

        Matching 2nd-order form with omega_n = 1/tau_w_s:
            Kp = 2*zeta*omega_n*J_total / Kt_eff
            Ki = omega_n^2 * J_total / Kt_eff
        """
        omega_n = 1.0 / tau_w_s
        Kp = 2.0 * target_zeta * omega_n * self._J_total / self._Kt_eff
        Ki = omega_n ** 2 * self._J_total / self._Kt_eff
        return Kp, Ki

    def _tune_ziegler_nichols(self, Ts_s: float) -> tuple[float, float]:
        """
        Find ultimate gain with full speed plant (including current loop lag).

        Speed plant with 1.5-sample PWM delay and current loop lag:
            G_full(s) = (Kt_eff/J_total) / (s * (tau_cl*s + 1))
                        * (1 - 0.75*Ts*s) / (1 + 0.75*Ts*s)   [Padé delay]

        Phase crossover → omega_u, Tu.
        Z-N PI:
            Kp = 0.45 * Ku,  Ki = Kp * 1.2 / Tu
        """
        # Speed plant: Kt_eff/(J_total) / (s * (tau_cl*s + 1))
        # Represent as: (Kt_eff/J_total) / (tau_cl*s^2 + s)
        Kg = self._Kt_eff / self._J_total
        num_plant = np.array([Kg])
        den_plant = np.array([self._tau_cl, 1.0, 0.0])

        # Padé delay
        T_half = 0.75 * Ts_s
        num_delay = np.array([-T_half, 1.0])
        den_delay = np.array([T_half, 1.0])

        num_comb = np.polymul(num_plant, num_delay)
        den_comb = np.polymul(den_plant, den_delay)

        w = np.logspace(0, 6, 20000)
        _, mag_db, phase_deg = signal.bode((num_comb, den_comb), w=w)

        phase_shifted = phase_deg + 180.0
        crossings = np.where(np.diff(np.sign(phase_shifted)))[0]

        if len(crossings) == 0:
            # No phase crossover — fall back to pole-zero method
            Kp_pz, Ki_pz = self._tune_pole_zero(_DEFAULT_TAU_CL_S)
            return 0.45 * Kp_pz, 0.45 * Ki_pz

        idx = crossings[0]
        omega_u = float(np.interp(
            0.0,
            [phase_shifted[idx], phase_shifted[idx + 1]],
            [w[idx], w[idx + 1]],
        ))
        Tu = 2.0 * math.pi / omega_u
        mag_at_u = float(np.interp(omega_u, w, mag_db))
        Ku = 10.0 ** (-mag_at_u / 20.0)

        Kp = 0.45 * Ku
        Ki = Kp * 1.2 / Tu
        return Kp, Ki

    # ------------------------------------------------------------------
    # Verification suite
    # ------------------------------------------------------------------

    def _verify(self, Kp: float, Ki: float, method: str) -> SpeedLoopResult:
        """
        Run the full verification suite and return a populated SpeedLoopResult.

        Speed plant TF (with current loop lag):
            G_w(s) = (Kt_eff/J_total) / (s * (tau_cl*s + 1))

        Open-loop with PI:
            L(s) = C(s) * G_w(s)
                 = (Kp*s + Ki) * Kt_eff/(J_total)
                   / (s^2 * (tau_cl*s + 1))

        Numerator of OL: Kt_eff/J_total * [Kp, Ki]
        Denominator of OL: tau_cl*s^3 + s^2 = [tau_cl, 1, 0, 0]
        """
        Kg = self._Kt_eff / self._J_total

        # Open-loop TF
        num_ol = Kg * np.array([Kp, Ki])
        den_ol = np.array([self._tau_cl, 1.0, 0.0, 0.0])

        # Closed-loop TF (by feedback formula)
        # CL num = OL num (same), CL den = OL den + OL num (with leading zeros matched)
        # Polynomial addition: den_cl = den_ol + [0, num_ol]
        # den_ol is degree 3: [tau_cl, 1, 0, 0]
        # num_ol is degree 1: [Kg*Kp, Kg*Ki] → padded to degree 3: [0, 0, Kg*Kp, Kg*Ki]
        num_ol_padded = np.concatenate([
            np.zeros(len(den_ol) - len(num_ol)), num_ol
        ])
        den_cl = den_ol + num_ol_padded
        num_cl = num_ol

        w = np.logspace(-1, 6, 10000)

        # --- Open-loop Bode ---
        w_ol, mag_db, phase_deg = signal.bode((num_ol, den_ol), w=w)

        crossover_Hz = 0.0
        PM_deg = 0.0
        gc_idx = np.where(np.diff(np.sign(mag_db)))[0]
        if len(gc_idx) > 0:
            i = gc_idx[0]
            w_c = float(np.interp(0.0, [mag_db[i], mag_db[i + 1]],
                                  [w_ol[i], w_ol[i + 1]]))
            crossover_Hz = w_c / (2.0 * math.pi)
            phase_at_c = float(np.interp(w_c, w_ol, phase_deg))
            PM_deg = 180.0 + phase_at_c

        GM_dB = 999.0
        phase_shifted = phase_deg + 180.0
        pc_idx = np.where(np.diff(np.sign(phase_shifted)))[0]
        if len(pc_idx) > 0:
            # Skip the integrator's -180 at DC (look for crossings above omega=1)
            valid = [i for i in pc_idx if w_ol[i] > 1.0]
            if valid:
                i = valid[0]
                w_p = float(np.interp(
                    0.0,
                    [phase_shifted[i], phase_shifted[i + 1]],
                    [w_ol[i], w_ol[i + 1]],
                ))
                mag_at_p = float(np.interp(w_p, w_ol, mag_db))
                GM_dB = -mag_at_p

        # --- Closed-loop BW (-3 dB) ---
        w_cl, mag_cl_db, _ = signal.bode((num_cl, den_cl), w=w)
        BW_Hz = 0.0
        # Find DC gain first (may not be at index 0 due to double integrator)
        # Use gain at lowest frequency as reference
        dc_db = mag_cl_db[0]
        bw_target = dc_db - 3.0
        bw_idx = np.where(np.diff(np.sign(mag_cl_db - bw_target)))[0]
        if len(bw_idx) > 0:
            i = bw_idx[0]
            w_bw = float(np.interp(
                bw_target,
                [mag_cl_db[i + 1], mag_cl_db[i]],
                [w_cl[i + 1], w_cl[i]],
            ))
            BW_Hz = w_bw / (2.0 * math.pi)

        # --- Step response ---
        t_end = max(0.2, 50 * self._tau_cl, 10 * (1.0 / (2.0 * math.pi * max(crossover_Hz, 1.0))))
        t = np.linspace(0, t_end, 10000)
        t_out, y_out = signal.step((num_cl, den_cl), T=t)
        y_final = float(y_out[-1])

        settling_ms = 0.0
        overshoot_pct = 0.0
        if abs(y_final) > 1e-10:
            band = _SETTLE_BAND * abs(y_final)
            outside = np.where(np.abs(y_out - y_final) > band)[0]
            if len(outside) > 0:
                settling_ms = float(t_out[outside[-1]]) * 1e3
            peak = float(np.max(y_out))
            if peak > y_final:
                overshoot_pct = (peak / y_final - 1.0) * 100.0

        # --- Warnings ---
        warns = []
        if PM_deg < _PM_MIN_DEG:
            warns.append(
                f"WARNING: Phase margin {PM_deg:.1f} deg < {_PM_MIN_DEG:.0f} deg"
            )
        if GM_dB < _GM_MIN_DB and GM_dB < 900:
            warns.append(
                f"WARNING: Gain margin {GM_dB:.1f} dB < {_GM_MIN_DB:.0f} dB"
            )
        if self.i_loop.BW_Hz > 0 and crossover_Hz > self.i_loop.BW_Hz / _BW_RATIO:
            warns.append(
                f"WARNING: Speed BW {crossover_Hz:.1f} Hz exceeds "
                f"current loop BW ({self.i_loop.BW_Hz:.1f} Hz) / {_BW_RATIO:.0f}"
            )

        # Anti-windup back-calculation gain
        Kb = Ki / Kp if Kp > 1e-12 else 0.0

        result = SpeedLoopResult(
            Kp=Kp, Ki=Ki, method=method, axis="speed",
            BW_Hz=BW_Hz, PM_deg=PM_deg, GM_dB=GM_dB,
            crossover_Hz=crossover_Hz,
            settling_ms=settling_ms, overshoot_pct=overshoot_pct,
            warnings=warns,
            Kb=Kb,
            Kt_eff=self._Kt_eff,
            J_total=self._J_total,
            B_eff=self._B_eff,
            tau_cl=self._tau_cl,
        )
        result._num_ol = num_ol
        result._den_ol = den_ol
        result._num_cl = num_cl
        result._den_cl = den_cl
        result._tau_plant = self._tau_cl
        return result

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_bode(self, result: SpeedLoopResult, save_path: str = None) -> None:
        """
        Bode plot of open-loop speed loop L(s).
        Crossover frequency marked. PM and GM annotated.
        """
        if result._num_ol is None:
            print("plot_bode: no TF data stored.")
            return

        w = np.logspace(-1, 6, 5000)
        w_ol, mag_db, phase_deg = signal.bode((result._num_ol, result._den_ol), w=w)
        f_hz = w_ol / (2 * np.pi)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(
            f"Bode — Speed Loop  [{result.method}]  "
            f"Kp={result.Kp:.4f}, Ki={result.Ki:.4f}"
        )

        ax1.semilogx(f_hz, mag_db, color="steelblue", linewidth=1.5)
        ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        if result.crossover_Hz > 0:
            ax1.axvline(result.crossover_Hz, color="crimson", linewidth=1,
                        linestyle="--", label=f"fc={result.crossover_Hz:.1f} Hz")
            ax1.legend(fontsize=9)
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, which="both", alpha=0.3)
        gm_str = f"{result.GM_dB:.1f}" if result.GM_dB < 900 else "inf"
        ax1.set_title(f"PM={result.PM_deg:.1f}  GM={gm_str} dB", fontsize=10)

        ax2.semilogx(f_hz, phase_deg, color="darkorange", linewidth=1.5)
        ax2.axhline(-180, color="gray", linewidth=0.8, linestyle="--")
        if result.crossover_Hz > 0:
            ax2.axvline(result.crossover_Hz, color="crimson", linewidth=1,
                        linestyle="--")
            ax2.annotate(
                f"PM={result.PM_deg:.1f}",
                xy=(result.crossover_Hz, -180 + result.PM_deg),
                xytext=(result.crossover_Hz * 3, -180 + result.PM_deg + 20),
                fontsize=8, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson"),
            )
        ax2.set_ylabel("Phase (deg)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        _save_or_show(fig, save_path)

    def plot_step(self, result: SpeedLoopResult, save_path: str = None) -> None:
        """
        Step response of closed-loop speed loop T(s).
        2% settling band shown as dashed lines.
        """
        if result._num_cl is None:
            print("plot_step: no TF data stored.")
            return

        t_end = max(result.settling_ms * 1e-3 * 10, 0.5)
        t = np.linspace(0, t_end, 8000)
        t_out, y_out = signal.step((result._num_cl, result._den_cl), T=t)
        y_final = float(y_out[-1])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_out * 1e3, y_out, color="steelblue", linewidth=1.5,
                label="Step response")
        ax.axhline(y_final, color="gray", linewidth=0.8, linestyle="--")
        if abs(y_final) > 1e-10:
            ax.axhline(y_final * 1.02, color="green", linewidth=0.8,
                       linestyle=":", label="+-2% band")
            ax.axhline(y_final * 0.98, color="green", linewidth=0.8,
                       linestyle=":")
        if result.settling_ms > 0:
            ax.axvline(result.settling_ms, color="crimson", linewidth=1,
                       linestyle="--",
                       label=f"Tset={result.settling_ms:.1f} ms")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Speed (normalized)")
        ax.set_title(
            f"Step Response — Speed Loop  [{result.method}]\n"
            f"Overshoot={result.overshoot_pct:.1f}%  Tsettle={result.settling_ms:.1f} ms"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, save_path)

    def plot_antiwindup(self, result: SpeedLoopResult, save_path: str = None) -> None:
        """
        Simulate anti-windup effect: step with and without output saturation.

        Shows back-calculation anti-windup recovering faster after saturation.
        Uses a simple Euler integration of the PI controller with saturation.
        """
        dt = 1e-4
        t_end = result.settling_ms * 1e-3 * 15 if result.settling_ms > 0 else 1.0
        t_end = max(t_end, 0.5)
        N = int(t_end / dt)
        t_arr = np.arange(N) * dt

        Kp, Ki, Kb = result.Kp, result.Ki, result.Kb
        Kg = self._Kt_eff / self._J_total

        # Saturation limits (representative: ±rated torque current)
        u_max = Kg * 10.0  # rough scale
        u_max = max(u_max, 1.0)

        def _simulate(antiwindup: bool):
            speed = 0.0
            integral = 0.0
            y_arr = np.zeros(N)
            for k in range(N):
                ref = 1.0
                err = ref - speed
                u_unsat = Kp * err + Ki * integral
                u_sat = float(np.clip(u_unsat, -u_max, u_max))
                # Speed update (simplified 1st-order plant)
                speed += dt * (Kg * u_sat - self._B_eff / self._J_total * speed)
                # Integrator update
                if antiwindup:
                    integral += dt * (err + Kb * (u_sat - u_unsat))
                else:
                    integral += dt * err
                y_arr[k] = speed
            return y_arr

        y_no_aw = _simulate(antiwindup=False)
        y_with_aw = _simulate(antiwindup=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_arr * 1e3, y_no_aw, color="crimson", linewidth=1.5,
                linestyle="--", label="No anti-windup")
        ax.plot(t_arr * 1e3, y_with_aw, color="steelblue", linewidth=1.5,
                label="Back-calculation AW")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Speed (normalized)")
        ax.set_title(
            f"Anti-Windup Comparison — Speed Loop  [{result.method}]  "
            f"Kb={result.Kb:.4f}"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path: str = None) -> None:
    """Save figure to file or display interactively."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
