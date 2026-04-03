"""
modules/position_loop.py — Position loop tuner for PMSM servo drive.

Two tuning methods:
  1. P  — proportional only; Kp_pos = BW_pos (rad/s)
  2. PD — adds derivative for damping improvement

Plant model (closed speed loop approximated as first-order lag):
    G_speed_cl(s) ≈ 1 / (tau_w*s + 1)   where tau_w = 1/(2*pi*BW_speed)
    G_pos(s) = G_speed_cl(s) / s = 1 / (s*(tau_w*s + 1))

P controller:
    C(s) = Kp_pos
    L(s) = Kp_pos / (s*(tau_w*s + 1))
    BW_pos (rad/s) ≈ Kp_pos  [valid when Kp_pos << 1/tau_w]
    Constraint: BW_pos <= BW_speed / 10

PD controller (velocity feedforward equivalent):
    C(s) = Kp_pos * (1 + Kd_pos*s)
    Kd_pos = 2*zeta / omega_n,  omega_n = 2*pi*BW_pos,  default zeta = 0.7

Velocity feedforward:
    Kff_v = 1.0 → zero steady-state following error for ramp inputs
    Kff_v = 0.0 → e_ss = 1/Kp_pos for unit ramp

Settling analysis:
    Step input: scipy.signal.step → 2% settling time
    Ramp input: steady-state following error = 1/Kp_pos  (Kff_v = 0)
                                             = 0          (Kff_v = 1)
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
_BW_RATIO: float = 10.0         # position BW must be <= speed loop BW / 10
_SETTLE_BAND: float = 0.02      # 2% settling band
_DEFAULT_ZETA: float = 0.7      # default damping ratio for PD design


# ---------------------------------------------------------------------------
# PositionLoopResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class PositionLoopResult(LoopResult):
    """
    Tuning result for the position loop.

    Extends LoopResult (Kp, Ki, BW_Hz, PM_deg, …) with position-specific fields.
    For a P controller Ki=0, Kd=0.
    For a PD controller Ki=0, Kd carries the derivative gain.
    """
    Kd: float = 0.0                   # derivative gain (PD only)
    Kff_v: float = 0.0                # velocity feedforward gain
    BW_speed_Hz: float = 0.0          # inner speed loop BW (Hz)
    following_error_ramp: float = 0.0 # e_ss for unit ramp [rad], Kff_v=0

    def summary(self) -> None:
        """Print a compact summary table to stdout."""
        print(
            f"\n{'='*70}\n"
            f"  Position Loop — method={self.method}\n"
            f"{'='*70}\n"
            f"  Kp_pos        = {self.Kp:.4f}  (rad/s per rad)\n"
            f"  Kd_pos        = {self.Kd:.4f}  (s)   [0 for P]\n"
            f"  Kff_v         = {self.Kff_v:.3f}  (velocity feedforward)\n"
            f"  BW_pos        = {self.BW_Hz:.2f} Hz  ({2*math.pi*self.BW_Hz:.1f} rad/s)\n"
            f"  BW_speed      = {self.BW_speed_Hz:.2f} Hz  (inner loop)\n"
            f"  PM            = {self.PM_deg:.1f} deg\n"
            f"  GM            = {self.GM_dB:.1f} dB\n"
            f"  settling      = {self.settling_ms:.1f} ms  (2% criterion)\n"
            f"  overshoot     = {self.overshoot_pct:.1f} %\n"
            f"  e_ss (ramp)   = {self.following_error_ramp:.4f} rad  (Kff_v=0)\n"
        )
        for w in self.warnings:
            print(f"  *** {w}")
        print("=" * 70)

    def plot_bode(self) -> None:
        """Bode plot of the open-loop TF with PM/GM annotations."""
        w, mag, phase = signal.bode(
            signal.TransferFunction(self._num_ol, self._den_ol),
            w=np.logspace(-1, 4, 2000)
        )
        freq_hz = w / (2 * math.pi)
        fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax_mag.semilogx(freq_hz, mag)
        ax_mag.axhline(0, color='k', lw=0.8, ls='--')
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.set_title(f"Position Loop Bode — {self.method}")
        ax_mag.grid(True, which='both')

        ax_ph.semilogx(freq_hz, phase)
        ax_ph.axhline(-180, color='k', lw=0.8, ls='--')
        ax_ph.set_xlabel("Frequency (Hz)")
        ax_ph.set_ylabel("Phase (deg)")
        ax_ph.grid(True, which='both')

        if self.crossover_Hz > 0:
            for ax in (ax_mag, ax_ph):
                ax.axvline(self.crossover_Hz, color='r', ls=':', lw=1.2,
                           label=f"crossover {self.crossover_Hz:.1f} Hz")
            ax_mag.legend()
            ax_mag.annotate(f"PM={self.PM_deg:.1f}°  GM={self.GM_dB:.1f} dB",
                            xy=(0.02, 0.05), xycoords='axes fraction', fontsize=9)
        plt.tight_layout()
        plt.show()

    def plot_step(self) -> None:
        """Closed-loop step response with 2% settling band."""
        sys_cl = signal.TransferFunction(self._num_cl, self._den_cl)
        t_end = max(0.5, 5 * self.settling_ms * 1e-3) if self.settling_ms > 0 else 0.5
        t = np.linspace(0, t_end, 5000)
        _, y = signal.step(sys_cl, T=t)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t * 1e3, y, label="Position step response")
        ax.axhline(1.0, color='k', lw=0.8, ls='--')
        ax.axhline(1.02, color='g', lw=0.8, ls='--', label="±2% band")
        ax.axhline(0.98, color='g', lw=0.8, ls='--')
        if self.settling_ms > 0:
            ax.axvline(self.settling_ms, color='r', ls=':', label=f"settling {self.settling_ms:.1f} ms")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Position (normalised)")
        ax.set_title(f"Position Loop Step — {self.method}")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# PositionLoopTuner
# ---------------------------------------------------------------------------

class PositionLoopTuner:
    """
    Outer position loop tuner.

    Only meaningful when the load config has ``position_loop_active: true``
    (i.e. load_position_servo.yaml). Call tune() to obtain a PositionLoopResult.

    Parameters
    ----------
    plant : PMSMPlant
        The PMSM plant (used for load type check only; position plant is
        derived from the closed speed loop).
    w_loop : LoopResult
        Closed-speed-loop result. BW_speed_Hz must be available.
    """

    def __init__(self, plant: PMSMPlant, w_loop: LoopResult) -> None:
        self.plant = plant
        self.w_loop = w_loop

        # Speed-loop closed-loop time constant
        self._tau_w: float = 1.0 / (2 * math.pi * w_loop.BW_Hz)  # [s]

        # Max allowable position BW
        self._bw_pos_max_hz: float = w_loop.BW_Hz / _BW_RATIO

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def tune(
        self,
        method: str = 'P',
        BW_pos_Hz: float | None = None,
        zeta: float = _DEFAULT_ZETA,
        Kff_v: float = 1.0,
    ) -> PositionLoopResult:
        """
        Tune the position loop.

        Parameters
        ----------
        method : str
            ``'P'`` or ``'PD'``.
        BW_pos_Hz : float, optional
            Desired position-loop bandwidth in Hz.
            Defaults to ``BW_speed / 10`` (the maximum stable value).
        zeta : float
            Damping ratio for PD design (default 0.7).
        Kff_v : float
            Velocity feedforward gain: 1.0 = perfect ramp tracking,
            0.0 = no feedforward.

        Returns
        -------
        PositionLoopResult
        """
        if BW_pos_Hz is None:
            BW_pos_Hz = self._bw_pos_max_hz

        if BW_pos_Hz > self._bw_pos_max_hz:
            raise ValueError(
                f"BW_pos_Hz={BW_pos_Hz:.2f} exceeds maximum "
                f"{self._bw_pos_max_hz:.2f} Hz (= BW_speed / {_BW_RATIO:.0f})"
            )

        if method == 'P':
            Kp, Kd = self._design_P(BW_pos_Hz)
        elif method == 'PD':
            Kp, Kd = self._design_PD(BW_pos_Hz, zeta)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'P' or 'PD'.")

        return self._verify(Kp, Kd, Kff_v, BW_pos_Hz, method)

    # ------------------------------------------------------------------
    # Private: design methods
    # ------------------------------------------------------------------

    def _design_P(self, BW_pos_Hz: float) -> tuple[float, float]:
        """
        P controller: Kp_pos = omega_c * sqrt(1 + (tau_w*omega_c)^2)

        Derived by solving |L(j*omega_c)| = 1 with L(s) = Kp_pos/(s*(tau_w*s+1)).

        At omega_c:
            |L| = Kp_pos / (omega_c * sqrt(1 + (tau_w*omega_c)^2)) = 1
            => Kp_pos = omega_c * sqrt(1 + (tau_w*omega_c)^2)
        """
        omega_c = 2 * math.pi * BW_pos_Hz
        Kp = omega_c * math.sqrt(1.0 + (self._tau_w * omega_c) ** 2)
        return Kp, 0.0

    def _design_PD(
        self, BW_pos_Hz: float, zeta: float
    ) -> tuple[float, float]:
        """
        PD controller: C(s) = Kp_pos * (1 + Kd_pos*s)

        Design equations (crossover pinned at omega_c = 2*pi*BW_pos_Hz):
          Kd_pos = 2*zeta / omega_c
          Kp_pos = omega_c * sqrt(1 + (tau_w*omega_c)^2) / sqrt(1 + (Kd*omega_c)^2)

        Derivation: set |L(j*omega_c)| = 1 with L = Kp*(Kd*s+1)/(s*(tau_w*s+1)):
          Kp * sqrt(1+(Kd*omega_c)^2) / (omega_c * sqrt(1+(tau_w*omega_c)^2)) = 1
          => Kp = omega_c * sqrt(1+(tau_w*omega_c)^2) / sqrt(1+(Kd*omega_c)^2)

        The PD zero at s = -1/Kd adds phase lead at crossover without
        shifting the gain crossover frequency, improving PM vs plain P.
        """
        omega_c = 2 * math.pi * BW_pos_Hz
        Kd = 2.0 * zeta / omega_c
        Kp = (omega_c * math.sqrt(1.0 + (self._tau_w * omega_c) ** 2)
              / math.sqrt(1.0 + (Kd * omega_c) ** 2))
        return Kp, Kd

    # ------------------------------------------------------------------
    # Private: verification suite
    # ------------------------------------------------------------------

    def _verify(
        self,
        Kp: float,
        Kd: float,
        Kff_v: float,
        BW_pos_Hz: float,
        method: str,
    ) -> PositionLoopResult:
        """
        Compute PM, GM, BW, settling, following error and build PositionLoopResult.

        Open-loop:
          P:  num=[Kp],       den=[tau_w, 1, 0]
          PD: num=[Kp*Kd, Kp], den=[tau_w, 1, 0]

        Closed-loop:
          T(s) = L(s)/(1+L(s))
          P:  num=[Kp],        den=[tau_w, 1, Kp]
          PD: num=[Kp*Kd, Kp], den=[tau_w, 1+Kp*Kd, Kp]
        """
        tau_w = self._tau_w

        # Open-loop numerator / denominator
        if Kd == 0.0:
            num_ol = np.array([Kp])
            num_cl = np.array([Kp])
            den_cl = np.array([tau_w, 1.0, Kp])
        else:
            num_ol = np.array([Kp * Kd, Kp])
            num_cl = np.array([Kp * Kd, Kp])
            den_cl = np.array([tau_w, 1.0 + Kp * Kd, Kp])
        den_ol = np.array([tau_w, 1.0, 0.0])

        sys_ol = signal.TransferFunction(num_ol, den_ol)

        # ---- Bode analysis ----
        w_range = np.logspace(-1, 5, 5000)
        w_out, H = signal.freqs(num_ol, den_ol, worN=w_range)
        mag_db = 20.0 * np.log10(np.abs(H) + 1e-300)
        phase_deg = np.degrees(np.unwrap(np.angle(H)))

        # Gain crossover: |L(jw)| = 0 dB
        gc_idx = np.where(np.diff(np.sign(mag_db)))[0]
        if gc_idx.size > 0:
            omega_c = float(w_out[gc_idx[0]])
            PM_deg = 180.0 + float(phase_deg[gc_idx[0]])
        else:
            omega_c = 0.0
            PM_deg = float('inf')

        # Phase crossover: phase(L(jw)) = -180 deg → gain margin
        pc_idx = np.where(np.diff(np.sign(phase_deg + 180.0)))[0]
        if pc_idx.size > 0:
            omega_pc = float(w_out[pc_idx[0]])
            GM_dB = -float(mag_db[pc_idx[0]])
        else:
            omega_pc = float('inf')
            GM_dB = float('inf')

        crossover_Hz = omega_c / (2.0 * math.pi)

        # ---- Closed-loop step response ----
        sys_cl = signal.TransferFunction(num_cl, den_cl)
        t_sim = np.linspace(0, max(0.2, 20.0 / (2 * math.pi * BW_pos_Hz)), 10000)
        _, y = signal.step(sys_cl, T=t_sim)

        settling_ms = _settling_time(t_sim, y, band=_SETTLE_BAND) * 1e3
        overshoot_pct = max(0.0, (np.max(y) - 1.0) * 100.0)

        # ---- Following error (ramp, Kff_v=0) ----
        # e_ss = 1/Kp_pos  (velocity error constant = Kp_pos for integrator plant)
        following_error = 1.0 / Kp if Kp > 0 else float('inf')

        # ---- Warnings ----
        warnings: list[str] = []
        if PM_deg < _PM_MIN_DEG:
            warnings.append(f"WARNING: Phase margin {PM_deg:.1f} deg < {_PM_MIN_DEG:.0f} deg")
        if GM_dB < _GM_MIN_DB:
            warnings.append(f"WARNING: Gain margin {GM_dB:.1f} dB < {_GM_MIN_DB:.0f} dB")
        if crossover_Hz > self._bw_pos_max_hz:
            warnings.append(
                f"WARNING: BW_pos {crossover_Hz:.2f} Hz > BW_speed/{_BW_RATIO:.0f} "
                f"= {self._bw_pos_max_hz:.2f} Hz"
            )

        return PositionLoopResult(
            Kp=Kp,
            Ki=0.0,
            method=method,
            axis='position',
            BW_Hz=crossover_Hz,
            PM_deg=PM_deg,
            GM_dB=GM_dB,
            crossover_Hz=crossover_Hz,
            settling_ms=settling_ms,
            overshoot_pct=overshoot_pct,
            warnings=warnings,
            Kd=Kd,
            Kff_v=Kff_v,
            BW_speed_Hz=self.w_loop.BW_Hz,
            following_error_ramp=following_error,
            _num_ol=num_ol,
            _den_ol=den_ol,
            _num_cl=num_cl,
            _den_cl=den_cl,
            _tau_plant=tau_w,
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _settling_time(t: np.ndarray, y: np.ndarray, band: float = 0.02) -> float:
    """
    Return the 2% settling time.

    Scans backwards from the end: the settling time is the last time the
    response leaves the ±band around the final value.
    """
    final = y[-1] if len(y) > 0 else 1.0
    if abs(final) < 1e-12:
        return 0.0
    outside = np.where(np.abs(y - final) > band * abs(final))[0]
    if len(outside) == 0:
        return 0.0
    return float(t[outside[-1]])
