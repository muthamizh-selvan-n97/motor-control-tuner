"""
modules/robustness.py — Robustness analysis for PMSM PI loops.

Three analyses:
  1. parameter_sweep  — PM and GM grids over Rs and L variations with FIXED gains
  2. sensitivity      — S(s) = 1/(1+L(s)) and T(s) = L/(1+L), peak sensitivity Ms
  3. margin_waterfall — PM and GM vs Rs deviation for current and speed loops

Typical variations modelled:
  - Rs drifts ±30% (temperature: +50% hot, -10% cold)
  - L varies ±20% (saturation at high current)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import signal

from modules.plant import PMSMPlant
from modules.current_loop import LoopResult

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_PM_MIN_DEG: float = 45.0       # stability margin threshold (deg)
_GM_MIN_DB: float = 6.0         # stability margin threshold (dB)
_MS_MAX: float = 2.0            # maximum peak sensitivity (6 dB)


# ---------------------------------------------------------------------------
# MarginPoint — single (Rs_delta, L_delta) result
# ---------------------------------------------------------------------------

@dataclass
class MarginPoint:
    Rs_delta: float   # fractional deviation, e.g. +0.3 = +30%
    L_delta: float    # fractional deviation, e.g. -0.2 = -20%
    PM_deg: float
    GM_dB: float


# ---------------------------------------------------------------------------
# RobustnessAnalyser
# ---------------------------------------------------------------------------

class RobustnessAnalyser:
    """
    Analyses how gain/phase margins change when motor parameters vary.

    Parameters
    ----------
    plant : PMSMPlant
        Nominal PMSM plant.
    i_loop : LoopResult
        Tuned current loop (gains FIXED during sweep).
    w_loop : LoopResult
        Tuned speed loop (gains FIXED during sweep).
    """

    def __init__(
        self,
        plant: PMSMPlant,
        i_loop: LoopResult,
        w_loop: LoopResult,
    ) -> None:
        self.plant = plant
        self.i_loop = i_loop
        self.w_loop = w_loop

        # Nominal parameters
        self._Rs_nom: float = plant.Rs
        self._Ld_nom: float = plant.Ld
        self._Lq_nom: float = plant.Lq

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------

    def parameter_sweep(
        self,
        Rs_range: tuple[float, float] = (-0.3, 0.3),
        L_range: tuple[float, float] = (-0.2, 0.2),
        steps: int = 21,
        axis: str = "d",
        save_path: str = None,
    ) -> dict:
        """
        Sweep Rs ± Rs_range and L (Ld or Lq) ± L_range simultaneously.

        For each (Rs_delta, L_delta) point, recomputes PM and GM using the
        FIXED gains from i_loop (no re-tuning).

        Parameters
        ----------
        Rs_range : (min_delta, max_delta)
            Fractional deviation range for Rs. E.g. (-0.3, 0.3) = ±30%.
        L_range  : (min_delta, max_delta)
            Fractional deviation range for L. E.g. (-0.2, 0.2) = ±20%.
        steps : int
            Number of grid points per axis.
        axis : str
            'd' or 'q' — which current loop to analyse.
        save_path : str, optional
            If given, saves the contour plot there.

        Returns
        -------
        dict with keys:
            'Rs_deltas'   : 1-D array of Rs fractional deviations
            'L_deltas'    : 1-D array of L fractional deviations
            'PM_grid'     : 2-D array [Rs_idx, L_idx] of phase margins (deg)
            'GM_grid'     : 2-D array [Rs_idx, L_idx] of gain margins (dB)
            'nominal_PM'  : float — PM at (0, 0)
            'nominal_GM'  : float
        """
        Rs_deltas = np.linspace(Rs_range[0], Rs_range[1], steps)
        L_deltas = np.linspace(L_range[0], L_range[1], steps)

        PM_grid = np.zeros((steps, steps))
        GM_grid = np.zeros((steps, steps))

        Kp = self.i_loop.Kp
        Ki = self.i_loop.Ki

        for i, dRs in enumerate(Rs_deltas):
            for j, dL in enumerate(L_deltas):
                Rs_v = self._Rs_nom * (1.0 + dRs)
                if axis == "d":
                    L_v = self._Ld_nom * (1.0 + dL)
                else:
                    L_v = self._Lq_nom * (1.0 + dL)
                tau_v = L_v / Rs_v

                # Open-loop TF with varied plant
                num_ol = np.array([Kp, Ki])
                den_ol = np.array([Rs_v * tau_v, Rs_v, 0.0])

                PM_v, GM_v = _compute_margins(num_ol, den_ol)
                PM_grid[i, j] = PM_v
                GM_grid[i, j] = GM_v

        result = {
            "Rs_deltas": Rs_deltas,
            "L_deltas": L_deltas,
            "PM_grid": PM_grid,
            "GM_grid": GM_grid,
            "nominal_PM": float(PM_grid[steps // 2, steps // 2]),
            "nominal_GM": float(GM_grid[steps // 2, steps // 2]),
        }

        self._plot_sweep(result, axis=axis, save_path=save_path)
        return result

    # ------------------------------------------------------------------
    # Sensitivity
    # ------------------------------------------------------------------

    def sensitivity(
        self,
        axis: str = "d",
        save_path: str = None,
    ) -> dict:
        """
        Compute sensitivity S(jω) and complementary sensitivity T(jω).

        S(s) = 1 / (1 + L(s))    — disturbance rejection
        T(s) = L(s) / (1 + L(s)) — reference tracking

        L(s) uses the current loop open-loop TF from i_loop._num_ol/_den_ol.

        Returns
        -------
        dict with keys:
            'freq_Hz'     : frequency array (Hz)
            'S_db'        : |S(jω)| in dB
            'T_db'        : |T(jω)| in dB
            'Ms_dB'       : peak sensitivity max|S| in dB
            'Mt_dB'       : peak complementary sensitivity max|T| in dB
            'Ms_Hz'       : frequency of peak |S|
            'warnings'    : list of warning strings
        """
        if self.i_loop._num_ol is None:
            raise RuntimeError("i_loop has no stored TF — run tuner._verify first.")

        num_ol = self.i_loop._num_ol
        den_ol = self.i_loop._den_ol

        w = np.logspace(0, 7, 5000)
        _, mag_db_ol, phase_ol = signal.bode((num_ol, den_ol), w=w)

        # Convert OL gain to linear
        mag_lin = 10.0 ** (mag_db_ol / 20.0)
        phase_rad = np.deg2rad(phase_ol)
        L_real = mag_lin * np.cos(phase_rad)
        L_imag = mag_lin * np.sin(phase_rad)
        L_cplx = L_real + 1j * L_imag

        S_cplx = 1.0 / (1.0 + L_cplx)
        T_cplx = L_cplx / (1.0 + L_cplx)

        S_db = 20.0 * np.log10(np.abs(S_cplx) + 1e-30)
        T_db = 20.0 * np.log10(np.abs(T_cplx) + 1e-30)
        f_hz = w / (2.0 * math.pi)

        Ms_dB = float(np.max(S_db))
        Mt_dB = float(np.max(T_db))
        Ms_Hz = float(f_hz[np.argmax(S_db)])

        warns = []
        if 10.0 ** (Ms_dB / 20.0) > _MS_MAX:
            warns.append(
                f"WARNING: Peak sensitivity Ms = {Ms_dB:.1f} dB "
                f"> {20*math.log10(_MS_MAX):.0f} dB — loop may be insufficiently robust"
            )

        result = {
            "freq_Hz": f_hz,
            "S_db": S_db,
            "T_db": T_db,
            "Ms_dB": Ms_dB,
            "Mt_dB": Mt_dB,
            "Ms_Hz": Ms_Hz,
            "warnings": warns,
        }

        self._plot_sensitivity(result, axis=axis, save_path=save_path)
        return result

    # ------------------------------------------------------------------
    # Margin waterfall
    # ------------------------------------------------------------------

    def margin_waterfall(
        self,
        Rs_range: tuple[float, float] = (-0.3, 0.3),
        steps: int = 31,
        axis: str = "d",
        save_path: str = None,
    ) -> dict:
        """
        PM and GM vs Rs deviation for current and speed loops (fixed gains).

        Plots both loops on the same figure with horizontal dashed lines at
        PM=45 deg and GM=6 dB.

        Parameters
        ----------
        Rs_range : (min_delta, max_delta)
            Fractional Rs deviation range.
        steps : int
            Number of points in the sweep.
        axis : str
            Current loop axis ('d' or 'q').
        save_path : str, optional
            Save path for the figure.

        Returns
        -------
        dict with keys:
            'Rs_deltas'   : 1-D array
            'PM_current'  : 1-D array of current loop PM (deg)
            'GM_current'  : 1-D array of current loop GM (dB)
            'PM_speed'    : 1-D array of speed loop PM (deg) [None if no w_loop TF]
            'GM_speed'    : 1-D array of speed loop GM (dB)
        """
        Rs_deltas = np.linspace(Rs_range[0], Rs_range[1], steps)

        PM_curr = np.zeros(steps)
        GM_curr = np.zeros(steps)
        PM_speed = np.zeros(steps)
        GM_speed = np.zeros(steps)

        Kp_i = self.i_loop.Kp
        Ki_i = self.i_loop.Ki
        L_nom = self._Ld_nom if axis == "d" else self._Lq_nom

        has_speed = (
            self.w_loop is not None
            and self.w_loop._num_ol is not None
        )
        Kp_w = self.w_loop.Kp if has_speed else 0.0
        Ki_w = self.w_loop.Ki if has_speed else 0.0

        # Speed plant parameters
        J_total = (
            getattr(self.w_loop, "J_total", None) or self.plant.J_total
        )
        Kt_eff = getattr(self.w_loop, "Kt_eff", None)
        if Kt_eff is None:
            Kt_eff = 1.5 * self.plant.p * self.plant.psi_f
        tau_cl = getattr(self.w_loop, "tau_cl", None)
        if tau_cl is None:
            tau_cl = 1.0 / (2.0 * math.pi * max(self.i_loop.BW_Hz, 1.0))
        Kg = Kt_eff / J_total

        for k, dRs in enumerate(Rs_deltas):
            Rs_v = self._Rs_nom * (1.0 + dRs)
            tau_v = L_nom / Rs_v

            # Current loop OL
            num_i = np.array([Kp_i, Ki_i])
            den_i = np.array([Rs_v * tau_v, Rs_v, 0.0])
            PM_curr[k], GM_curr[k] = _compute_margins(num_i, den_i)

            # Speed loop OL (only Rs affects Kt_eff scaling here; simplified)
            if has_speed:
                # tau_cl stays fixed (current gains fixed), Kg stays fixed
                num_w = Kg * np.array([Kp_w, Ki_w])
                den_w = np.array([tau_cl, 1.0, 0.0, 0.0])
                PM_speed[k], GM_speed[k] = _compute_margins_speed(num_w, den_w)

        result = {
            "Rs_deltas": Rs_deltas,
            "PM_current": PM_curr,
            "GM_current": GM_curr,
            "PM_speed": PM_speed if has_speed else None,
            "GM_speed": GM_speed if has_speed else None,
        }

        self._plot_waterfall(result, save_path=save_path)
        return result

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_sweep(self, result: dict, axis: str, save_path: str = None) -> None:
        """Contour plot of PM and GM over Rs vs L variation grid."""
        Rs_pct = result["Rs_deltas"] * 100.0
        L_pct = result["L_deltas"] * 100.0
        PM_grid = result["PM_grid"]
        GM_grid = result["GM_grid"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Robustness: PM and GM vs Rs/L variation  [{axis}-axis]",
            fontsize=12,
        )

        # PM contour
        cset1 = ax1.contourf(L_pct, Rs_pct, PM_grid, levels=20, cmap="RdYlGn")
        ax1.contour(L_pct, Rs_pct, PM_grid, levels=[_PM_MIN_DEG],
                    colors="black", linewidths=2, linestyles="--")
        fig.colorbar(cset1, ax=ax1, label="Phase margin (deg)")
        ax1.set_xlabel("L deviation (%)")
        ax1.set_ylabel("Rs deviation (%)")
        ax1.set_title(f"Phase Margin (dashed = {_PM_MIN_DEG:.0f} deg)")
        ax1.plot(0, 0, "w*", markersize=12, label="Nominal")
        ax1.legend(fontsize=9)

        # GM contour
        GM_plot = np.where(GM_grid > 50, 50, GM_grid)  # cap for plotting
        cset2 = ax2.contourf(L_pct, Rs_pct, GM_plot, levels=20, cmap="RdYlGn")
        ax2.contour(L_pct, Rs_pct, GM_grid, levels=[_GM_MIN_DB],
                    colors="black", linewidths=2, linestyles="--")
        fig.colorbar(cset2, ax=ax2, label="Gain margin (dB)")
        ax2.set_xlabel("L deviation (%)")
        ax2.set_ylabel("Rs deviation (%)")
        ax2.set_title(f"Gain Margin (dashed = {_GM_MIN_DB:.0f} dB)")
        ax2.plot(0, 0, "w*", markersize=12, label="Nominal")
        ax2.legend(fontsize=9)

        plt.tight_layout()
        _save_or_show(fig, save_path)

    def _plot_sensitivity(self, result: dict, axis: str, save_path: str = None) -> None:
        """Plot |S(jω)| and |T(jω)| vs frequency."""
        f_hz = result["freq_Hz"]
        S_db = result["S_db"]
        T_db = result["T_db"]
        Ms_dB = result["Ms_dB"]
        Mt_dB = result["Mt_dB"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogx(f_hz, S_db, color="steelblue", linewidth=1.5, label="|S(jω)|")
        ax.semilogx(f_hz, T_db, color="darkorange", linewidth=1.5, label="|T(jω)|")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(20 * math.log10(_MS_MAX), color="crimson", linewidth=1,
                   linestyle=":", label=f"Ms limit = {20*math.log10(_MS_MAX):.0f} dB")
        ax.annotate(
            f"Ms = {Ms_dB:.1f} dB @ {result['Ms_Hz']:.0f} Hz",
            xy=(result["Ms_Hz"], Ms_dB),
            xytext=(result["Ms_Hz"] * 3, Ms_dB + 2),
            fontsize=8, color="steelblue",
            arrowprops=dict(arrowstyle="->", color="steelblue"),
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(
            f"Sensitivity Functions — Current Loop ({axis}-axis)\n"
            f"Ms = {Ms_dB:.1f} dB   Mt = {Mt_dB:.1f} dB"
        )
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, save_path)

    def _plot_waterfall(self, result: dict, save_path: str = None) -> None:
        """PM and GM vs Rs deviation waterfall for current and speed loops."""
        Rs_pct = result["Rs_deltas"] * 100.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Margin Waterfall: PM and GM vs Rs deviation", fontsize=12)

        # --- Phase margin ---
        ax1.plot(Rs_pct, result["PM_current"], color="steelblue",
                 linewidth=1.5, label="Current loop")
        if result["PM_speed"] is not None:
            ax1.plot(Rs_pct, result["PM_speed"], color="darkorange",
                     linewidth=1.5, label="Speed loop")
        ax1.axhline(_PM_MIN_DEG, color="crimson", linewidth=1.5,
                    linestyle="--", label=f"PM = {_PM_MIN_DEG:.0f} deg")
        ax1.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax1.set_xlabel("Rs deviation (%)")
        ax1.set_ylabel("Phase margin (deg)")
        ax1.set_title("Phase Margin vs Rs variation")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)

        # --- Gain margin ---
        GM_curr_plot = np.where(result["GM_current"] > 80, 80,
                                result["GM_current"])
        ax2.plot(Rs_pct, GM_curr_plot, color="steelblue",
                 linewidth=1.5, label="Current loop")
        if result["GM_speed"] is not None:
            GM_speed_plot = np.where(result["GM_speed"] > 80, 80,
                                     result["GM_speed"])
            ax2.plot(Rs_pct, GM_speed_plot, color="darkorange",
                     linewidth=1.5, label="Speed loop")
        ax2.axhline(_GM_MIN_DB, color="crimson", linewidth=1.5,
                    linestyle="--", label=f"GM = {_GM_MIN_DB:.0f} dB")
        ax2.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax2.set_xlabel("Rs deviation (%)")
        ax2.set_ylabel("Gain margin (dB)")
        ax2.set_title("Gain Margin vs Rs variation")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)

        plt.tight_layout()
        _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_margins(num_ol: np.ndarray, den_ol: np.ndarray) -> tuple[float, float]:
    """
    Compute (PM_deg, GM_dB) from open-loop TF numerator/denominator.

    Uses a dense log-spaced frequency grid for reliable interpolation.
    Returns (0, 999) if crossover / phase crossover not found.
    """
    w = np.logspace(0, 7, 8000)
    _, mag_db, phase_deg = signal.bode((num_ol, den_ol), w=w)

    PM_deg = 0.0
    gc_idx = np.where(np.diff(np.sign(mag_db)))[0]
    if len(gc_idx) > 0:
        i = gc_idx[0]
        w_c = float(np.interp(0.0, [mag_db[i], mag_db[i + 1]],
                              [w[i], w[i + 1]]))
        phase_at_c = float(np.interp(w_c, w, phase_deg))
        PM_deg = 180.0 + phase_at_c

    GM_dB = 999.0
    phase_shifted = phase_deg + 180.0
    pc_idx = np.where(np.diff(np.sign(phase_shifted)))[0]
    if len(pc_idx) > 0:
        i = pc_idx[0]
        w_p = float(np.interp(0.0, [phase_shifted[i], phase_shifted[i + 1]],
                               [w[i], w[i + 1]]))
        GM_dB = -float(np.interp(w_p, w, mag_db))

    return PM_deg, GM_dB


def _compute_margins_speed(num_ol: np.ndarray, den_ol: np.ndarray) -> tuple[float, float]:
    """
    Same as _compute_margins but skips the integrator phase crossing at DC.
    The speed plant has a double-integrator, so phase starts at -180 at low w.
    """
    w = np.logspace(-1, 5, 10000)
    _, mag_db, phase_deg = signal.bode((num_ol, den_ol), w=w)

    PM_deg = 0.0
    gc_idx = np.where(np.diff(np.sign(mag_db)))[0]
    if len(gc_idx) > 0:
        i = gc_idx[0]
        w_c = float(np.interp(0.0, [mag_db[i], mag_db[i + 1]],
                              [w[i], w[i + 1]]))
        phase_at_c = float(np.interp(w_c, w, phase_deg))
        PM_deg = 180.0 + phase_at_c

    GM_dB = 999.0
    phase_shifted = phase_deg + 180.0
    # Skip low-frequency crossing (integrator artefact, w < 1 rad/s)
    pc_idx = [i for i in np.where(np.diff(np.sign(phase_shifted)))[0]
              if w[i] > 1.0]
    if len(pc_idx) > 0:
        i = pc_idx[0]
        w_p = float(np.interp(0.0, [phase_shifted[i], phase_shifted[i + 1]],
                               [w[i], w[i + 1]]))
        GM_dB = -float(np.interp(w_p, w, mag_db))

    return PM_deg, GM_dB


def _save_or_show(fig, save_path: str = None) -> None:
    """Save figure to file or display interactively."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    elif matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)
