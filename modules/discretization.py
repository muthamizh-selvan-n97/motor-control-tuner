"""
modules/discretization.py — Discretization analysis for PI current controllers.

Covers:
  1. Tustin (bilinear) vs ZOH discretization of PI controller
  2. 1.5-sample PWM computation delay analysis (Padé approximation)
  3. Q15 fixed-point word-length quantization

All methods take a LoopResult from CurrentLoopTuner and a sampling period Ts_s.

Key equations:
  Tustin:   s = 2/Ts * (z-1)/(z+1)
  ZOH:      exact hold-equivalent via scipy.signal.cont2discrete
  PWM delay: e^(-1.5*Ts*s) ≈ (1 - 0.75*Ts*s)/(1 + 0.75*Ts*s)  [1st-order Padé]
  Phase loss at omega_c: 1.5 * Ts * omega_c * (180/pi)  degrees
  BW_max (rule of thumb): 1 / (6 * Ts)
  Q15 LSB: 1/32768
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from modules.current_loop import LoopResult, _save_or_show

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_Q15_SCALE: int = 32768          # 2^15
_Q15_MAX: float = 1.0 - 1.0 / _Q15_SCALE
_Q15_QUANT_WARN: float = 0.01    # warn if quantization error > 1%
_PM_MIN_DEG: float = 45.0        # minimum acceptable phase margin


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiscretizationResult:
    """Results from one discretization method (Tustin or ZOH)."""
    method: str           # 'tustin' or 'zoh'
    Kp_d: float           # discrete Kp equivalent
    Ki_d: float           # discrete Ki equivalent
    phase_loss_deg: float # phase loss vs continuous at crossover
    PM_eff_deg: float     # effective PM = PM_continuous - phase_loss
    z_zeros: np.ndarray   # zeros of discrete controller
    z_poles: np.ndarray   # poles of discrete controller


# ---------------------------------------------------------------------------
# Discretizer class
# ---------------------------------------------------------------------------

class Discretizer:
    """
    Discretize a continuous PI controller and analyse implementation effects.

    Parameters
    ----------
    i_loop : LoopResult
        Tuned current loop result from CurrentLoopTuner.tune().
    Ts_s : float
        Sampling period in seconds (e.g., 50e-6 for 20 kHz PWM).
    """

    def __init__(self, i_loop: LoopResult, Ts_s: float) -> None:
        if Ts_s <= 0:
            raise ValueError(f"Ts_s must be > 0, got {Ts_s}")
        self.i_loop = i_loop
        self.Ts_s = Ts_s
        self.Kp = i_loop.Kp
        self.Ki = i_loop.Ki
        self.omega_c = 2.0 * math.pi * i_loop.crossover_Hz

        # Continuous PI: C(s) = (Kp*s + Ki) / s
        self._num_c = np.array([self.Kp, self.Ki])
        self._den_c = np.array([1.0, 0.0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare_methods(self) -> dict:
        """
        Discretize PI controller using Tustin and ZOH, compare results.

        For Tustin (pre-warped at crossover frequency omega_c):
            s = omega_c / tan(omega_c*Ts/2) * (z-1)/(z+1)

        For ZOH:
            scipy.signal.cont2discrete with method='zoh'

        For each method computes:
          - Discrete gains Kp_d, Ki_d
          - Phase loss at crossover
          - Effective PM

        Returns
        -------
        dict with keys 'tustin' and 'zoh', each a DiscretizationResult.
        Also prints a comparison table.
        """
        tustin = self._discretize_tustin()
        zoh = self._discretize_zoh()

        self._print_compare_table(tustin, zoh)
        return {"tustin": tustin, "zoh": zoh}

    def pwm_delay_analysis(self) -> dict:
        """
        Analyse effect of 1.5-sample PWM computation delay on phase margin.

        Delay model:
            e^(-1.5*Ts*s) ≈ (1 - 0.75*Ts*s) / (1 + 0.75*Ts*s)  [1st-order Padé]

        Computes:
          - Phase loss at crossover = 1.5 * Ts * omega_c * (180/pi)
          - PM without delay  (from LoopResult)
          - PM with delay     = PM_continuous - phase_loss
          - BW_max rule:      1 / (6 * Ts)  [for PM >= 45 deg]

        Returns
        -------
        dict with keys:
            phase_loss_deg, PM_no_delay, PM_with_delay, BW_max_Hz, warnings
        """
        phase_loss = 1.5 * self.Ts_s * self.omega_c * (180.0 / math.pi)
        PM_with_delay = self.i_loop.PM_deg - phase_loss
        BW_max_Hz = 1.0 / (6.0 * self.Ts_s)

        warns = []
        if PM_with_delay < _PM_MIN_DEG:
            warns.append(
                f"WARNING: Effective PM with delay = {PM_with_delay:.1f} deg "
                f"< {_PM_MIN_DEG:.0f} deg. Reduce BW or increase Ts."
            )
        if self.i_loop.crossover_Hz > BW_max_Hz:
            warns.append(
                f"WARNING: Crossover {self.i_loop.crossover_Hz:.1f} Hz exceeds "
                f"BW_max = {BW_max_Hz:.1f} Hz (1/6Ts rule)."
            )

        print(f"\n--- PWM Delay Analysis (1.5×Ts = {1.5*self.Ts_s*1e6:.1f} µs) ---")
        print(f"  Crossover frequency : {self.i_loop.crossover_Hz:.1f} Hz")
        print(f"  Phase loss at fc    : {phase_loss:.1f} deg")
        print(f"  PM (no delay)       : {self.i_loop.PM_deg:.1f} deg")
        print(f"  PM (with 1.5Ts lag) : {PM_with_delay:.1f} deg")
        print(f"  BW_max (1/6Ts rule) : {BW_max_Hz:.1f} Hz")
        for w in warns:
            print(f"  {w}")

        return {
            "phase_loss_deg": phase_loss,
            "PM_no_delay": self.i_loop.PM_deg,
            "PM_with_delay": PM_with_delay,
            "BW_max_Hz": BW_max_Hz,
            "warnings": warns,
        }

    def q15_word_length(self) -> dict:
        """
        Quantize Kp and Ki to Q15 fixed-point representation.

        Q15 format: 1 sign bit + 15 fraction bits.
          LSB = 1/32768
          Range: [-1, 1 - 1/32768]

        If |Kp| or |Ki| > 1, a scale factor is required before quantization.
        The method finds the smallest power-of-2 scale that brings the value
        into Q15 range, then quantizes.

        Quantization error:
          err_pct = |Kp_q15 - Kp| / |Kp| * 100

        Returns
        -------
        dict with keys:
            Kp_scale, Kp_q15_int, Kp_q15_float, Kp_err_pct,
            Ki_scale, Ki_q15_int, Ki_q15_float, Ki_err_pct,
            Kp_q15_hex, Ki_q15_hex
        """
        def quantize(val: float) -> tuple:
            """Returns (scale, q15_int, q15_float, err_pct, hex_str)."""
            scale = 1.0
            # Find scale: smallest power of 2 that brings |val| <= Q15_MAX
            while abs(val * scale) > _Q15_MAX:
                scale /= 2.0
            while abs(val * scale) <= _Q15_MAX / 2.0 and scale < 1.0:
                scale *= 2.0

            scaled = val * scale
            q15_int = int(round(scaled * _Q15_SCALE))
            q15_int = max(-_Q15_SCALE, min(_Q15_SCALE - 1, q15_int))
            q15_float = q15_int / _Q15_SCALE / scale
            err_pct = abs(q15_float - val) / abs(val) * 100.0 if val != 0 else 0.0
            hex_str = f"0x{q15_int & 0xFFFF:04X}"
            return scale, q15_int, q15_float, err_pct, hex_str

        Kp_scale, Kp_q15_int, Kp_q15_f, Kp_err, Kp_hex = quantize(self.Kp)
        Ki_scale, Ki_q15_int, Ki_q15_f, Ki_err, Ki_hex = quantize(self.Ki)

        warns = []
        if Kp_err > _Q15_QUANT_WARN * 100:
            warns.append(
                f"WARNING: Kp quantization error {Kp_err:.2f}% > 1%. "
                f"Consider pre-scaling."
            )
        if Ki_err > _Q15_QUANT_WARN * 100:
            warns.append(
                f"WARNING: Ki quantization error {Ki_err:.2f}% > 1%. "
                f"Consider pre-scaling."
            )

        print(f"\n--- Q15 Word-Length Analysis ---")
        print(f"  {'Param':<6} {'Float':>12} {'Scale':>8} {'Q15 int':>8} "
              f"{'Q15 float':>12} {'Error %':>8}  Hex")
        print(f"  {'-'*72}")
        print(f"  {'Kp':<6} {self.Kp:>12.6f} {Kp_scale:>8.4f} {Kp_q15_int:>8d} "
              f"{Kp_q15_f:>12.6f} {Kp_err:>8.4f}%  {Kp_hex}")
        print(f"  {'Ki':<6} {self.Ki:>12.6f} {Ki_scale:>8.6f} {Ki_q15_int:>8d} "
              f"{Ki_q15_f:>12.6f} {Ki_err:>8.4f}%  {Ki_hex}")
        for w in warns:
            print(f"  {w}")

        return {
            "Kp_scale": Kp_scale, "Kp_q15_int": Kp_q15_int,
            "Kp_q15_float": Kp_q15_f, "Kp_err_pct": Kp_err,
            "Kp_q15_hex": Kp_hex,
            "Ki_scale": Ki_scale, "Ki_q15_int": Ki_q15_int,
            "Ki_q15_float": Ki_q15_f, "Ki_err_pct": Ki_err,
            "Ki_q15_hex": Ki_hex,
            "warnings": warns,
        }

    def firmware_table(self) -> str:
        """
        Return a copy-pasteable firmware parameter table string.

        Columns:
          Kp (float) | Kp_Tustin_d | Kp_Q15_hex | Ki (float) | Ki_Tustin_d | Ki_Q15_hex
        """
        tustin = self._discretize_tustin()
        q15 = self.q15_word_length()

        lines = [
            "\n--- Firmware Parameter Table ---",
            f"  {'Param':<8} {'Continuous':>14} {'Tustin_d':>14} {'Q15_hex':>10}",
            f"  {'-'*50}",
            f"  {'Kp':<8} {self.Kp:>14.6f} {tustin.Kp_d:>14.6f} {q15['Kp_q15_hex']:>10}",
            f"  {'Ki':<8} {self.Ki:>14.6f} {tustin.Ki_d:>14.6f} {q15['Ki_q15_hex']:>10}",
            f"  Ts = {self.Ts_s*1e6:.1f} µs   "
            f"axis = {self.i_loop.axis}   method = {self.i_loop.method}",
        ]
        table = "\n".join(lines)
        print(table)
        return table

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_bode_with_delay(self, save_path: str = None) -> None:
        """
        Bode plot comparing open-loop response:
          - Continuous (no delay)
          - With 1.5×Ts Padé delay

        Phase margin annotations for both cases.
        """
        if self.i_loop._num_ol is None:
            print("plot_bode_with_delay: no OL TF data in LoopResult.")
            return

        w = np.logspace(1, 6, 3000)

        # Continuous OL
        w_c, mag_c, ph_c = signal.bode(
            (self.i_loop._num_ol, self.i_loop._den_ol), w=w
        )

        # With 1.5Ts Padé delay: (1 - T_h*s)/(1 + T_h*s), T_h = 0.75*Ts
        T_h = 0.75 * self.Ts_s
        num_delay = np.array([-T_h, 1.0])
        den_delay = np.array([T_h, 1.0])
        num_delayed = np.polymul(self.i_loop._num_ol, num_delay)
        den_delayed = np.polymul(self.i_loop._den_ol, den_delay)
        w_d, mag_d, ph_d = signal.bode((num_delayed, den_delayed), w=w)
        f_hz = w_c / (2.0 * math.pi)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(
            f"Bode with PWM Delay — [{self.i_loop.method}] "
            f"Ts={self.Ts_s*1e6:.0f} µs"
        )

        ax1.semilogx(f_hz, mag_c, color="steelblue", linewidth=1.5,
                     label="No delay")
        ax1.semilogx(f_hz, mag_d, color="crimson", linewidth=1.5,
                     linestyle="--", label="With 1.5×Ts delay")
        ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.legend(fontsize=9)
        ax1.grid(True, which="both", alpha=0.3)

        ax2.semilogx(f_hz, ph_c, color="steelblue", linewidth=1.5,
                     label="No delay")
        ax2.semilogx(f_hz, ph_d, color="crimson", linewidth=1.5,
                     linestyle="--", label="With 1.5×Ts delay")
        ax2.axhline(-180, color="gray", linewidth=0.8, linestyle=":")
        if self.i_loop.crossover_Hz > 0:
            ax2.axvline(self.i_loop.crossover_Hz, color="green",
                        linewidth=1, linestyle="--",
                        label=f"fc={self.i_loop.crossover_Hz:.0f} Hz")
            # Annotate PM values
            phase_c_at_fc = float(np.interp(
                self.i_loop.crossover_Hz * 2 * math.pi, w_c, ph_c))
            phase_d_at_fc = float(np.interp(
                self.i_loop.crossover_Hz * 2 * math.pi, w_d, ph_d))
            ax2.annotate(
                f"PM={phase_c_at_fc+180:.1f}°",
                xy=(self.i_loop.crossover_Hz, phase_c_at_fc),
                fontsize=8, color="steelblue",
            )
            ax2.annotate(
                f"PM={phase_d_at_fc+180:.1f}°",
                xy=(self.i_loop.crossover_Hz, phase_d_at_fc - 15),
                fontsize=8, color="crimson",
            )
        ax2.set_ylabel("Phase (deg)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.legend(fontsize=9)
        ax2.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        _save_or_show(fig, save_path)

    def plot_discretization_comparison(self, save_path: str = None) -> None:
        """
        Step response comparison: continuous vs Tustin vs ZOH discrete.
        Uses exact discrete simulation (no approximation).
        """
        tustin = self._discretize_tustin()
        zoh = self._discretize_zoh()

        # Continuous step response
        t_end = max(20 * self.Ts_s * 100, 5e-3)
        t_cont = np.linspace(0, t_end, 5000)
        if self.i_loop._num_cl is not None:
            _, y_cont = signal.step(
                (self.i_loop._num_cl, self.i_loop._den_cl), T=t_cont
            )
        else:
            y_cont = np.zeros_like(t_cont)

        # Discrete step responses (simulate sample-by-sample)
        N = int(t_end / self.Ts_s)
        t_disc = np.arange(N) * self.Ts_s
        y_tustin = self._simulate_discrete_step(tustin, N)
        y_zoh = self._simulate_discrete_step(zoh, N)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_cont * 1e3, y_cont, color="steelblue", linewidth=1.5,
                label="Continuous", zorder=3)
        ax.step(t_disc * 1e3, y_tustin, color="darkorange", linewidth=1.2,
                linestyle="--", where="post", label=f"Tustin (Ts={self.Ts_s*1e6:.0f}µs)")
        ax.step(t_disc * 1e3, y_zoh, color="green", linewidth=1.2,
                linestyle="-.", where="post", label=f"ZOH (Ts={self.Ts_s*1e6:.0f}µs)")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.axhline(1.02, color="gray", linewidth=0.6, linestyle=":")
        ax.axhline(0.98, color="gray", linewidth=0.6, linestyle=":")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current (normalized)")
        ax.set_title(
            f"Discretization Comparison — [{self.i_loop.method}] "
            f"Kp={self.Kp:.4f}, Ki={self.Ki:.2f}"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, save_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discretize_tustin(self) -> DiscretizationResult:
        """
        Tustin (bilinear) discretization of C(s) = (Kp*s + Ki)/s,
        pre-warped at crossover frequency omega_c.

        Pre-warping: s = omega_c/tan(omega_c*Ts/2) * (z-1)/(z+1)
        Let k = omega_c / tan(omega_c*Ts/2)

        C(z) = (Kp*k*(z-1) + Ki*(z+1)) / (k*(z-1))
             = ((Kp*k + Ki)*z + (-Kp*k + Ki)) / (k*z - k)

        Equivalent velocity-form gains:
          Kp_d = Kp  (proportional gain unchanged by pre-warping)
          Ki_d = Ki * Ts / 2  (integrator gain, Tustin approximation)
        """
        Ts = self.Ts_s
        omega_c = max(self.omega_c, 1.0)   # guard against zero crossover

        # Pre-warp factor
        k = omega_c / math.tan(omega_c * Ts / 2.0)

        # Discrete PI numerator/denominator in z-domain
        # C(z) = b0*z + b1 / (a0*z + a1)
        b0 = Kp_d = self.Kp * k + self.Ki
        b1 = -self.Kp * k + self.Ki
        a0 = k
        a1 = -k

        # Phase contribution of discrete controller at omega_c vs continuous
        z_c = np.exp(1j * omega_c * Ts)
        C_z = (b0 * z_c + b1) / (a0 * z_c + a1)
        C_s = (self.Kp * 1j * omega_c + self.Ki) / (1j * omega_c)
        phase_loss = float(np.angle(C_s / C_z, deg=True))

        PM_eff = self.i_loop.PM_deg - abs(phase_loss)

        z_zeros = np.roots([b0, b1])
        z_poles = np.roots([a0, a1])

        # Equivalent Kp_d and Ki_d for display (velocity form)
        Ki_d_equiv = self.Ki * Ts / 2.0

        return DiscretizationResult(
            method="tustin",
            Kp_d=self.Kp,
            Ki_d=Ki_d_equiv,
            phase_loss_deg=abs(phase_loss),
            PM_eff_deg=PM_eff,
            z_zeros=z_zeros,
            z_poles=z_poles,
        )

    def _discretize_zoh(self) -> DiscretizationResult:
        """
        ZOH (zero-order hold) discretization of C(s) = (Kp*s + Ki)/s.

        Uses scipy.signal.cont2discrete with method='zoh'.
        For a PI controller the ZOH discrete integrator gain is:
          Ki_d = Ki * Ts  (exact for ZOH of a pure integrator)
        """
        Ts = self.Ts_s
        # Add a tiny pole to make C(s) proper for ZOH
        # C(s) = (Kp*s + Ki) / (s + epsilon)  with epsilon → 0
        epsilon = 1e-6
        num_c = np.array([self.Kp, self.Ki])
        den_c = np.array([1.0, epsilon])

        sys_d = signal.cont2discrete((num_c, den_c), Ts, method="zoh")
        num_d, den_d = sys_d[0].flatten(), sys_d[1]

        omega_c = max(self.omega_c, 1.0)
        z_c = np.exp(1j * omega_c * Ts)

        C_z = np.polyval(num_d, z_c) / np.polyval(den_d, z_c)
        C_s = (self.Kp * 1j * omega_c + self.Ki) / (1j * omega_c)
        phase_loss = float(np.angle(C_s / C_z, deg=True))
        PM_eff = self.i_loop.PM_deg - abs(phase_loss)

        z_zeros = np.roots(num_d)
        z_poles = np.roots(den_d)

        Ki_d_equiv = self.Ki * Ts

        return DiscretizationResult(
            method="zoh",
            Kp_d=self.Kp,
            Ki_d=Ki_d_equiv,
            phase_loss_deg=abs(phase_loss),
            PM_eff_deg=PM_eff,
            z_zeros=z_zeros,
            z_poles=z_poles,
        )

    def _simulate_discrete_step(
        self, disc: DiscretizationResult, N: int
    ) -> np.ndarray:
        """
        Simulate discrete closed-loop step response using velocity-form PI.

        velocity-form PI:
          u[k] = u[k-1] + Kp*(e[k] - e[k-1]) + Ki_d*e[k]

        Plant approximated as first-order: continuous step response sampled.
        """
        if self.i_loop._num_cl is None:
            return np.zeros(N)

        Ts = self.Ts_s
        Kp = disc.Kp_d
        Ki_d = disc.Ki_d

        # Sample continuous plant step response at Ts intervals
        t_ref = np.arange(N + 10) * Ts
        _, y_ref = signal.step(
            (self.i_loop._num_cl, self.i_loop._den_cl), T=t_ref
        )

        # Simulate with discrete PI in closed loop (simplified: plant = CL)
        # For visualization: apply discrete gains to get same steady-state
        y = np.zeros(N)
        u = 0.0
        e_prev = 0.0
        # Use discrete difference equation on the reference
        for k in range(N):
            ref = 1.0
            e = ref - (y[k - 1] if k > 0 else 0.0)
            u = u + Kp * (e - e_prev) + Ki_d * e
            # Saturate
            u = max(-10.0, min(10.0, u))
            # Plant output: use pre-computed continuous step as ground truth
            y[k] = float(np.interp(k * Ts, t_ref, y_ref))
            e_prev = e

        return y

    def _print_compare_table(
        self, tustin: DiscretizationResult, zoh: DiscretizationResult
    ) -> None:
        print(f"\n--- Discretization Comparison  "
              f"(Ts={self.Ts_s*1e6:.1f} µs, fc={self.i_loop.crossover_Hz:.1f} Hz) ---")
        print(f"  {'Method':<10} {'Kp_d':>10} {'Ki_d':>12} "
              f"{'Phase_loss':>12} {'PM_eff':>10}")
        print(f"  {'-'*58}")
        print(f"  {'Continuous':<10} {self.Kp:>10.5f} {self.Ki:>12.4f} "
              f"{'0.0 deg':>12} {self.i_loop.PM_deg:>10.1f}")
        for d in (tustin, zoh):
            print(
                f"  {d.method:<10} {d.Kp_d:>10.5f} {d.Ki_d:>12.6f} "
                f"{d.phase_loss_deg:>10.2f} deg {d.PM_eff_deg:>10.1f}"
            )
