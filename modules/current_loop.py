"""
modules/current_loop.py — PI current loop tuner for PMSM d-q axes.

Four tuning methods (all produce the same LoopResult and verification suite):
  1. pole_zero        — analytical pole-zero cancellation
  2. frequency_domain — place gain crossover at desired BW
  3. root_locus       — place closed-loop poles at desired damping ratio
  4. ziegler_nichols  — simulate P-only with Padé delay, find Ku/Tu

All methods use the decoupled per-axis plant (cross-coupling treated as disturbance):
    G_d(s) = (1/Rs) / (tau_d*s + 1),  tau_d = Ld/Rs
    G_q(s) = (1/Rs) / (tau_q*s + 1),  tau_q = Lq/Rs

PI controller:
    C(s) = Kp*(s + Ki/Kp) / s = (Kp*s + Ki) / s

Open-loop:   L(s) = C(s)*G(s) = (Kp*s + Ki) / (Rs*s*(tau*s + 1))
Closed-loop: T(s) = L(s)/(1+L(s)) = (Kp*s + Ki) / (Rs*tau*s^2 + (Rs+Kp)*s + Ki)
DC gain of T: Ki/Ki = 1 (integral ensures unity steady-state gain)
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

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_PM_MIN_DEG: float = 45.0       # minimum acceptable phase margin
_GM_MIN_DB: float = 6.0         # minimum acceptable gain margin
_BW_RATIO: float = 10.0         # speed BW must be <= current BW / 10
_DEFAULT_TAU_CL_S: float = 1e-3  # default closed-loop time constant (1 ms)
_DEFAULT_BW_HZ: float = 500.0    # default crossover BW for freq-domain method
_DEFAULT_ZETA: float = 0.707     # default damping ratio for root-locus method
_DEFAULT_TS_S: float = 50e-6     # default PWM period (20 kHz) for Z-N method
_PADE_ORDER: float = 1.5         # 1.5-sample PWM delay
_SETTLE_BAND: float = 0.02       # 2% settling band


# ---------------------------------------------------------------------------
# LoopResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    """
    Tuning result and verification metrics for one PI loop.

    Produced by CurrentLoopTuner.tune() and SpeedLoopTuner.tune().
    """
    Kp: float
    Ki: float
    method: str
    axis: str               # 'd', 'q', or 'speed'

    # Verification suite (populated by _verify)
    BW_Hz: float = 0.0
    PM_deg: float = 0.0
    GM_dB: float = 0.0
    crossover_Hz: float = 0.0
    settling_ms: float = 0.0
    overshoot_pct: float = 0.0
    warnings: list = field(default_factory=list)

    # Private: TF coefficients stored for plotting (not shown in repr)
    _num_ol: np.ndarray = field(default=None, repr=False)
    _den_ol: np.ndarray = field(default=None, repr=False)
    _num_cl: np.ndarray = field(default=None, repr=False)
    _den_cl: np.ndarray = field(default=None, repr=False)
    _tau_plant: float = field(default=0.0, repr=False)

    def summary(self) -> None:
        """Print an 8-column summary table to stdout."""
        hdr = f"{'Method':<18} {'Axis':<5} {'Kp':>8} {'Ki':>8} " \
              f"{'BW_Hz':>7} {'PM_deg':>7} {'GM_dB':>7} {'Tset_ms':>8}"
        print(hdr)
        print("-" * len(hdr))
        gm_str = f"{self.GM_dB:>7.1f}" if self.GM_dB < 900 else f"{'inf':>7}"
        print(
            f"{self.method:<18} {self.axis:<5} {self.Kp:>8.4f} {self.Ki:>8.2f} "
            f"{self.BW_Hz:>7.1f} {self.PM_deg:>7.1f} {gm_str} "
            f"{self.settling_ms:>8.3f}"
        )
        for w in self.warnings:
            print(f"  {w}")

    def plot_bode(self, save_path: str = None) -> None:
        """
        Bode plot of open-loop L(s): magnitude (dB) and phase (deg).
        Crossover frequency marked. PM and GM annotated.
        """
        if self._num_ol is None:
            print("plot_bode: no TF data stored.")
            return

        w = np.logspace(0, 6, 2000)
        w_ol, mag_db, phase_deg = signal.bode((self._num_ol, self._den_ol), w=w)
        f_hz = w_ol / (2 * np.pi)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(
            f"Bode — Current Loop ({self.axis}-axis)  [{self.method}]  "
            f"Kp={self.Kp:.4f}, Ki={self.Ki:.2f}"
        )

        ax1.semilogx(f_hz, mag_db, color="steelblue", linewidth=1.5)
        ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        if self.crossover_Hz > 0:
            ax1.axvline(self.crossover_Hz, color="crimson", linewidth=1,
                        linestyle="--", label=f"fc={self.crossover_Hz:.1f} Hz")
            ax1.legend(fontsize=9)
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True, which="both", alpha=0.3)
        gm_str = f"{self.GM_dB:.1f}" if self.GM_dB < 900 else "inf"
        ax1.set_title(f"PM={self.PM_deg:.1f}°   GM={gm_str} dB", fontsize=10)

        ax2.semilogx(f_hz, phase_deg, color="darkorange", linewidth=1.5)
        ax2.axhline(-180, color="gray", linewidth=0.8, linestyle="--")
        if self.crossover_Hz > 0:
            ax2.axvline(self.crossover_Hz, color="crimson", linewidth=1,
                        linestyle="--")
            ax2.annotate(
                f"PM={self.PM_deg:.1f}°",
                xy=(self.crossover_Hz, -180 + self.PM_deg),
                xytext=(self.crossover_Hz * 2, -180 + self.PM_deg + 20),
                fontsize=8, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson"),
            )
        ax2.set_ylabel("Phase (deg)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        _save_or_show(fig, save_path)

    def plot_step(self, save_path: str = None) -> None:
        """
        Step response of closed-loop T(s).
        2% settling band shown as dashed lines.
        """
        if self._num_cl is None:
            print("plot_step: no TF data stored.")
            return

        t_end = max(10 * self.settling_ms * 1e-3, 20 * self._tau_plant, 0.05)
        t = np.linspace(0, t_end, 5000)
        t_out, y_out = signal.step((self._num_cl, self._den_cl), T=t)
        y_final = y_out[-1]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_out * 1e3, y_out, color="steelblue", linewidth=1.5,
                label="Step response")
        ax.axhline(y_final, color="gray", linewidth=0.8, linestyle="--")
        if abs(y_final) > 1e-10:
            ax.axhline(y_final * 1.02, color="green", linewidth=0.8,
                       linestyle=":", label="±2% band")
            ax.axhline(y_final * 0.98, color="green", linewidth=0.8,
                       linestyle=":")
        if self.settling_ms > 0:
            ax.axvline(self.settling_ms, color="crimson", linewidth=1,
                       linestyle="--",
                       label=f"Tset={self.settling_ms:.2f} ms")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current (normalized)")
        ax.set_title(
            f"Step Response — Current Loop ({self.axis}-axis)  [{self.method}]\n"
            f"Overshoot={self.overshoot_pct:.1f}%  Tsettle={self.settling_ms:.2f} ms"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, save_path)

    def plot_root_locus(self, save_path: str = None) -> None:
        """
        Root locus: closed-loop poles as Kp varies (Ki/Kp fixed at tuned ratio).
        Current operating point marked.
        """
        if self._num_cl is None:
            print("plot_root_locus: no TF data stored.")
            return

        tau = self._tau_plant
        Rs = self.Ki / self.Kp * tau if self.Kp > 0 else 1.0

        # Sweep Kp, keep Ki/Kp ratio fixed
        ratio = self.Ki / self.Kp if self.Kp > 0 else 1.0
        Kp_sweep = np.linspace(0.01 * self.Kp, 5.0 * self.Kp, 300)
        real_parts, imag_parts = [], []

        for Kp_k in Kp_sweep:
            Ki_k = Kp_k * ratio
            _, den_cl_k = _build_cl_tf(Kp_k, Ki_k, self._den_ol[0] / self._den_cl[0],
                                        tau)
            roots = np.roots(den_cl_k)
            for r in roots:
                real_parts.append(r.real)
                imag_parts.append(r.imag)

        # Current operating point poles
        cl_poles = np.roots(self._den_cl)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(real_parts, imag_parts, s=2, color="steelblue", alpha=0.4,
                   label="Root locus")
        ax.scatter(cl_poles.real, cl_poles.imag, s=60, color="crimson",
                   zorder=5, marker="x", label="Operating poles")
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        ax.set_title(
            f"Root Locus — Current Loop ({self.axis}-axis)  [{self.method}]"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# CurrentLoopTuner
# ---------------------------------------------------------------------------

class CurrentLoopTuner:
    """
    Tune PI current controllers for d and q axes using four methods.

    Parameters
    ----------
    plant : PMSMPlant
        PMSM plant model (used for motor type and structure).
    params : dict
        Identified parameters from ParameterIdentifier (Rs, Ld, Lq, psi_f).
        If None, falls back to plant electrical parameters.
    """

    def __init__(self, plant: PMSMPlant, params: dict = None) -> None:
        self.plant = plant

        if params is not None:
            self._Rs = float(params["Rs"])
            self._Ld = float(params["Ld"])
            self._Lq = float(params["Lq"])
        else:
            self._Rs = plant.Rs
            self._Ld = plant.Ld
            self._Lq = plant.Lq

        # For SPMSM, Lq = Ld
        if plant.motor_type == "SPMSM":
            self._Lq = self._Ld

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(
        self,
        method: str = "pole_zero",
        axis: str = "d",
        **kwargs,
    ) -> LoopResult:
        """
        Tune the current loop PI controller.

        Parameters
        ----------
        method : str
            'pole_zero', 'frequency_domain', 'root_locus', or 'ziegler_nichols'.
        axis : str
            'd' or 'q'.
        **kwargs
            Method-specific keyword arguments (see individual _tune_* methods).

        Returns
        -------
        LoopResult
            Gains plus full verification suite.
        """
        if axis not in ("d", "q"):
            raise ValueError(f"axis must be 'd' or 'q', got '{axis}'")

        if method == "pole_zero":
            tau_cl_s = kwargs.get("tau_cl_s", _DEFAULT_TAU_CL_S)
            Kp, Ki = self._tune_pole_zero(axis, tau_cl_s)

        elif method == "frequency_domain":
            BW_Hz = kwargs.get("BW_Hz", _DEFAULT_BW_HZ)
            Kp, Ki = self._tune_frequency_domain(axis, BW_Hz)

        elif method == "root_locus":
            target_zeta = kwargs.get("target_zeta", _DEFAULT_ZETA)
            tau_cl_s = kwargs.get("tau_cl_s", _DEFAULT_TAU_CL_S)
            Kp, Ki = self._tune_root_locus(axis, target_zeta, tau_cl_s)

        elif method == "ziegler_nichols":
            Ts_s = kwargs.get("Ts_s", _DEFAULT_TS_S)
            Kp, Ki = self._tune_ziegler_nichols(axis, Ts_s)

        else:
            raise ValueError(
                f"method must be one of 'pole_zero', 'frequency_domain', "
                f"'root_locus', 'ziegler_nichols'. Got '{method}'."
            )

        return self._verify(Kp, Ki, method, axis)

    # ------------------------------------------------------------------
    # Tuning methods
    # ------------------------------------------------------------------

    def _tune_pole_zero(self, axis: str, tau_cl_s: float) -> tuple[float, float]:
        """
        Pole-zero cancellation.

        Plant zero: s = -Rs/L  (plant pole location).
        PI zero placed at same location: Ki/Kp = Rs/L.
        Desired closed-loop pole: s = -1/tau_cl.

        Kp = L / tau_cl
        Ki = Rs / tau_cl

        The closed-loop transfer function becomes first-order:
            T(s) ≈ 1 / (tau_cl*s + 1)   [after zero-pole cancellation]
        """
        L = self._Ld if axis == "d" else self._Lq
        Kp = L / tau_cl_s
        Ki = self._Rs / tau_cl_s
        return Kp, Ki

    def _tune_frequency_domain(self, axis: str, BW_Hz: float) -> tuple[float, float]:
        """
        Place gain crossover at desired bandwidth.

        At omega_c = 2*pi*BW_Hz, |C(j*omega_c)*G(j*omega_c)| = 1.

        With Ki = Kp * omega_c / 10:
            |C(j*omega_c)| ≈ Kp * sqrt(1 + (1/10)^2) ≈ Kp * sqrt(1.01)
            |G_d(j*omega_c)| = (1/Rs) / sqrt(1 + (tau*omega_c)^2)

        Kp = Rs * sqrt(1 + (tau*omega_c)^2) / sqrt(1.01)
        Ki = Kp * omega_c / 10
        """
        L = self._Ld if axis == "d" else self._Lq
        tau = L / self._Rs
        omega_c = 2.0 * math.pi * BW_Hz
        Kp = self._Rs * math.sqrt(1.0 + (tau * omega_c) ** 2) / math.sqrt(1.01)
        Ki = Kp * omega_c / 10.0
        return Kp, Ki

    def _tune_root_locus(
        self, axis: str, target_zeta: float, tau_cl_s: float
    ) -> tuple[float, float]:
        """
        Place closed-loop poles at desired damping ratio.

        For PI + first-order plant, the closed-loop characteristic equation is:
            tau*s^2 + (1 + Kp/Rs)*s + Ki/Rs = 0

        Matching to standard 2nd-order form s^2 + 2*zeta*omega_n*s + omega_n^2 = 0:
            omega_n = 1 / tau_cl   (desired natural frequency)
            Ki = omega_n^2 * Rs * tau
            Kp = Rs * (2*zeta*omega_n*tau - 1)

        If Kp <= 0 (omega_n too low), omega_n is increased to the minimum value
        that yields Kp > 0: omega_n_min = 1 / (2*zeta*tau).
        """
        L = self._Ld if axis == "d" else self._Lq
        tau = L / self._Rs
        omega_n = 1.0 / tau_cl_s

        Kp = self._Rs * (2.0 * target_zeta * omega_n * tau - 1.0)
        if Kp <= 0:
            omega_n = 1.5 / (2.0 * target_zeta * tau)
            Kp = self._Rs * (2.0 * target_zeta * omega_n * tau - 1.0)

        Ki = omega_n ** 2 * self._Rs * tau
        return Kp, Ki

    def _tune_ziegler_nichols(self, axis: str, Ts_s: float) -> tuple[float, float]:
        """
        Find ultimate gain using 1.5-sample PWM delay (1st-order Padé approximation).

        Padé delay: e^(-1.5*Ts*s) ≈ (1 - 0.75*Ts*s) / (1 + 0.75*Ts*s)

        Plant with delay:
            G_delay(s) = G_plant(s) * delay(s)

        Phase crossover (angle = -180°) → ultimate frequency omega_u, period Tu.
        Ku = 1 / |G_delay(j*omega_u)|.

        Ziegler-Nichols PI formulas:
            Kp = 0.45 * Ku
            Ki = Kp / (Tu / 1.2) = Kp * 1.2 / Tu
        """
        L = self._Ld if axis == "d" else self._Lq
        tau = L / self._Rs

        # Plant TF
        num_p = np.array([1.0 / self._Rs])
        den_p = np.array([tau, 1.0])

        # 1st-order Padé for 1.5*Ts delay: (1 - T_half*s) / (1 + T_half*s)
        T_half = 0.75 * Ts_s
        num_delay = np.array([-T_half, 1.0])
        den_delay = np.array([T_half, 1.0])

        # Plant + delay
        num_comb = np.polymul(num_p, num_delay)
        den_comb = np.polymul(den_p, den_delay)

        w = np.logspace(2, 7, 20000)
        _, mag_db, phase_deg = signal.bode((num_comb, den_comb), w=w)

        # Find phase = -180°
        phase_shifted = phase_deg + 180.0
        crossings = np.where(np.diff(np.sign(phase_shifted)))[0]

        if len(crossings) == 0:
            # No phase crossover — fall back to pole-zero with conservative scaling
            Kp_pz, Ki_pz = self._tune_pole_zero(axis, tau_cl_s=_DEFAULT_TAU_CL_S)
            return 0.45 * Kp_pz, 0.45 * Ki_pz

        idx = crossings[0]
        omega_u = float(np.interp(
            0.0,
            [phase_shifted[idx], phase_shifted[idx + 1]],
            [w[idx], w[idx + 1]],
        ))
        Tu = 2.0 * math.pi / omega_u

        # Ku = 1 / |G_delay(j*omega_u)|
        mag_at_u = float(np.interp(omega_u, w, mag_db))
        Ku = 10.0 ** (-mag_at_u / 20.0)

        Kp = 0.45 * Ku
        Ki = Kp * 1.2 / Tu
        return Kp, Ki

    # ------------------------------------------------------------------
    # Verification suite
    # ------------------------------------------------------------------

    def _verify(
        self,
        Kp: float,
        Ki: float,
        method: str,
        axis: str,
        inner_loop_BW_Hz: float = None,
    ) -> LoopResult:
        """
        Run the full verification suite and return a populated LoopResult.

        Steps:
          1. Build open-loop and closed-loop TFs.
          2. Bode analysis → crossover_Hz, PM_deg, GM_dB.
          3. Closed-loop -3 dB bandwidth → BW_Hz.
          4. Step response → settling_ms (2% band), overshoot_pct.
          5. Warnings for PM < 45°, GM < 6 dB, BW hierarchy violation.
        """
        L = self._Ld if axis == "d" else self._Lq
        tau = L / self._Rs

        # Build TFs
        # OL: L(s) = (Kp*s + Ki) / (Rs*s*(tau*s + 1))
        num_ol = np.array([Kp, Ki])
        den_ol = np.array([self._Rs * tau, self._Rs, 0.0])

        # CL: T(s) = (Kp*s + Ki) / (Rs*tau*s^2 + (Rs+Kp)*s + Ki)
        num_cl = np.array([Kp, Ki])
        den_cl = np.array([self._Rs * tau, self._Rs + Kp, Ki])

        w = np.logspace(0, 7, 8000)

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

        GM_dB = 999.0  # effectively infinite
        phase_shifted = phase_deg + 180.0
        pc_idx = np.where(np.diff(np.sign(phase_shifted)))[0]
        if len(pc_idx) > 0:
            i = pc_idx[0]
            w_p = float(np.interp(0.0,
                                  [phase_shifted[i], phase_shifted[i + 1]],
                                  [w_ol[i], w_ol[i + 1]]))
            mag_at_p = float(np.interp(w_p, w_ol, mag_db))
            GM_dB = -mag_at_p

        # --- Closed-loop BW (-3 dB from DC) ---
        w_cl, mag_cl_db, _ = signal.bode((num_cl, den_cl), w=w)
        dc_db = mag_cl_db[0]
        bw_target = dc_db - 3.0
        bw_idx = np.where(np.diff(np.sign(mag_cl_db - bw_target)))[0]
        BW_Hz = 0.0
        if len(bw_idx) > 0:
            i = bw_idx[0]
            w_bw = float(np.interp(
                bw_target,
                [mag_cl_db[i + 1], mag_cl_db[i]],
                [w_cl[i + 1], w_cl[i]],
            ))
            BW_Hz = w_bw / (2.0 * math.pi)

        # --- Step response ---
        t_end = max(30 * tau, 5e-3, 20 * (1.0 / (2.0 * math.pi * BW_Hz + 1e-3)))
        t = np.linspace(0, t_end, 8000)
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
        if inner_loop_BW_Hz is not None and BW_Hz > inner_loop_BW_Hz / _BW_RATIO:
            warns.append(
                f"WARNING: BW {BW_Hz:.1f} Hz exceeds inner loop BW "
                f"({inner_loop_BW_Hz:.1f} Hz) / {_BW_RATIO:.0f}"
            )

        result = LoopResult(
            Kp=Kp, Ki=Ki, method=method, axis=axis,
            BW_Hz=BW_Hz, PM_deg=PM_deg, GM_dB=GM_dB,
            crossover_Hz=crossover_Hz,
            settling_ms=settling_ms, overshoot_pct=overshoot_pct,
            warnings=warns,
        )
        result._num_ol = num_ol
        result._den_ol = den_ol
        result._num_cl = num_cl
        result._den_cl = den_cl
        result._tau_plant = tau
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cl_tf(Kp: float, Ki: float, Rs_tau: float, tau: float):
    """Build closed-loop TF coefficients (used by root locus plot)."""
    Rs = Rs_tau / tau
    num_cl = np.array([Kp, Ki])
    den_cl = np.array([Rs * tau, Rs + Kp, Ki])
    return num_cl, den_cl


def _save_or_show(fig, save_path: str = None) -> None:
    """Save figure to file or display interactively."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
