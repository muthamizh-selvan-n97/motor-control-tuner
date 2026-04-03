"""
modules/dashboard.py — Combined 3x2 dashboard figure and CSV export.

Layout (3 rows x 2 columns):
  [0,0] Bode magnitude+phase — current loop (d-axis)
  [0,1] Bode magnitude+phase — speed loop
  [1,0] Step response — current loop (d + q overlay)
  [1,1] Step response — speed loop
  [2,0] Root locus — current loop (d-axis)
  [2,1] Robustness waterfall — PM vs Rs deviation

CSV columns:
  loop, method, Kp, Ki, BW_Hz, PM_deg, GM_dB, settling_ms, overshoot_pct, warnings
"""

from __future__ import annotations

import csv
import math
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from modules.current_loop import LoopResult
from modules.robustness import RobustnessAnalyser, _compute_margins, _compute_margins_speed
from modules.plant import PMSMPlant

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_PM_MIN_DEG: float = 45.0
_GM_MIN_DB: float = 6.0
_FIG_SIZE: tuple = (14, 10)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dashboard(
    params: dict,
    i_loop_d: LoopResult,
    i_loop_q: LoopResult,
    w_loop: LoopResult,
    plant: Optional[PMSMPlant] = None,
    save: bool = True,
    output_dir: str = "outputs/",
) -> None:
    """
    Generate a 3x2 dashboard figure and a CSV gains summary.

    Parameters
    ----------
    params : dict
        Identified motor parameters (Rs, Ld, Lq, psi_f).
    i_loop_d : LoopResult
        Tuned d-axis current loop result.
    i_loop_q : LoopResult
        Tuned q-axis current loop result.
    w_loop : LoopResult
        Tuned speed loop result (SpeedLoopResult).
    plant : PMSMPlant, optional
        Used for robustness waterfall. If None, the waterfall panel is skipped.
    save : bool
        If True, saves figure and CSV to output_dir.
    output_dir : str
        Directory for output files.
    """
    motor_name = _safe_name(
        getattr(i_loop_d, "_motor_name", None)
        or params.get("motor_name", "motor")
    )
    load_name = _safe_name(
        getattr(w_loop, "_load_name", None)
        or params.get("load_name", "load")
    )

    fig = plt.figure(figsize=_FIG_SIZE)
    fig.suptitle(
        f"Motor Control Tuner Dashboard — {motor_name} / {load_name}",
        fontsize=13,
        fontweight="bold",
    )

    gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.35)

    # ---- [0,0] Current loop Bode ----
    ax_bode_i_mag = fig.add_subplot(gs[0, 0])
    ax_bode_i_ph = ax_bode_i_mag.twinx()
    _plot_bode_compact(fig, gs[0, 0], i_loop_d,
                       title=f"Current Loop Bode (d-axis) [{i_loop_d.method}]")

    # ---- [0,1] Speed loop Bode ----
    _plot_bode_compact(fig, gs[0, 1], w_loop,
                       title=f"Speed Loop Bode [{w_loop.method}]",
                       is_speed=True)

    # ---- [1,0] Current loop step (d+q overlay) ----
    _plot_step_overlay(fig, gs[1, 0], i_loop_d, i_loop_q,
                       title="Step Response — Current Loop (d+q)")

    # ---- [1,1] Speed loop step ----
    _plot_step_single(fig, gs[1, 1], w_loop,
                      title=f"Step Response — Speed Loop [{w_loop.method}]",
                      ylabel="Speed (normalized)")

    # ---- [2,0] Root locus (current loop d-axis) ----
    _plot_root_locus(fig, gs[2, 0], i_loop_d,
                     title=f"Root Locus — Current Loop (d) [{i_loop_d.method}]")

    # ---- [2,1] Robustness waterfall ----
    if plant is not None:
        _plot_waterfall_compact(fig, gs[2, 1], plant, i_loop_d, w_loop,
                                title="PM vs Rs deviation")
    else:
        ax = fig.add_subplot(gs[2, 1])
        ax.text(0.5, 0.5, "Robustness waterfall\n(pass plant= to enable)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
                color="gray")
        ax.set_axis_off()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f"dashboard_{motor_name}_{load_name}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {fig_path}")

        csv_path = os.path.join(output_dir, f"gains_summary_{motor_name}_{load_name}.csv")
        _write_csv(csv_path, i_loop_d, i_loop_q, w_loop)
        print(f"Saved: {csv_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _plot_bode_compact(
    fig: plt.Figure,
    gs_loc,
    loop: LoopResult,
    title: str,
    is_speed: bool = False,
) -> None:
    """Two-subplot Bode (magnitude + phase) sharing x-axis, packed into one gridspec cell."""
    inner = gs_loc.subgridspec(2, 1, hspace=0.05)
    ax_mag = fig.add_subplot(inner[0])
    ax_ph = fig.add_subplot(inner[1], sharex=ax_mag)

    if loop._num_ol is None:
        ax_mag.text(0.5, 0.5, "No TF data", ha="center", va="center",
                    transform=ax_mag.transAxes)
        return

    w = np.logspace(-1 if is_speed else 0, 6, 3000)
    _, mag_db, phase_deg = signal.bode((loop._num_ol, loop._den_ol), w=w)
    f_hz = w / (2.0 * math.pi)

    ax_mag.semilogx(f_hz, mag_db, color="steelblue", linewidth=1.2)
    ax_mag.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    if loop.crossover_Hz > 0:
        ax_mag.axvline(loop.crossover_Hz, color="crimson", linewidth=0.9,
                       linestyle="--", alpha=0.8)
    gm_str = f"{loop.GM_dB:.1f}" if loop.GM_dB < 900 else "inf"
    ax_mag.set_title(
        f"{title}\nPM={loop.PM_deg:.1f}°  GM={gm_str} dB  fc={loop.crossover_Hz:.1f} Hz",
        fontsize=8,
    )
    ax_mag.set_ylabel("dB", fontsize=7)
    ax_mag.tick_params(labelbottom=False, labelsize=7)
    ax_mag.grid(True, which="both", alpha=0.25)

    ax_ph.semilogx(f_hz, phase_deg, color="darkorange", linewidth=1.2)
    ax_ph.axhline(-180, color="gray", linewidth=0.7, linestyle="--")
    if loop.crossover_Hz > 0:
        ax_ph.axvline(loop.crossover_Hz, color="crimson", linewidth=0.9,
                      linestyle="--", alpha=0.8)
    ax_ph.set_ylabel("deg", fontsize=7)
    ax_ph.set_xlabel("Hz", fontsize=7)
    ax_ph.tick_params(labelsize=7)
    ax_ph.grid(True, which="both", alpha=0.25)


def _plot_step_overlay(
    fig: plt.Figure,
    gs_loc,
    loop_d: LoopResult,
    loop_q: LoopResult,
    title: str,
) -> None:
    """Overlay d-axis and q-axis current step responses."""
    ax = fig.add_subplot(gs_loc)
    ax.set_title(title, fontsize=8)

    for loop, color, label in [
        (loop_d, "steelblue", f"d-axis ({loop_d.method})"),
        (loop_q, "darkorange", f"q-axis ({loop_q.method})"),
    ]:
        if loop._num_cl is None:
            continue
        tau = loop._tau_plant
        t_end = max(10 * loop.settling_ms * 1e-3, 10 * tau, 0.05)
        t = np.linspace(0, t_end, 3000)
        t_out, y_out = signal.step((loop._num_cl, loop._den_cl), T=t)
        y_final = float(y_out[-1])
        ax.plot(t_out * 1e3, y_out, color=color, linewidth=1.2, label=label)
        if abs(y_final) > 1e-10:
            ax.axhline(y_final * 1.02, color=color, linewidth=0.6,
                       linestyle=":", alpha=0.6)
            ax.axhline(y_final * 0.98, color=color, linewidth=0.6,
                       linestyle=":", alpha=0.6)

    ax.set_xlabel("Time (ms)", fontsize=7)
    ax.set_ylabel("Current (norm.)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)


def _plot_step_single(
    fig: plt.Figure,
    gs_loc,
    loop: LoopResult,
    title: str,
    ylabel: str = "Output (norm.)",
) -> None:
    """Single step response panel."""
    ax = fig.add_subplot(gs_loc)
    ax.set_title(title, fontsize=8)

    if loop._num_cl is None:
        ax.text(0.5, 0.5, "No TF data", ha="center", va="center",
                transform=ax.transAxes)
        return

    t_end = max(loop.settling_ms * 1e-3 * 8, 0.5)
    t = np.linspace(0, t_end, 4000)
    t_out, y_out = signal.step((loop._num_cl, loop._den_cl), T=t)
    y_final = float(y_out[-1])

    ax.plot(t_out * 1e3, y_out, color="steelblue", linewidth=1.2)
    ax.axhline(y_final, color="gray", linewidth=0.7, linestyle="--")
    if abs(y_final) > 1e-10:
        ax.axhline(y_final * 1.02, color="green", linewidth=0.6,
                   linestyle=":", label="+-2%")
        ax.axhline(y_final * 0.98, color="green", linewidth=0.6,
                   linestyle=":")
    if loop.settling_ms > 0:
        ax.axvline(loop.settling_ms, color="crimson", linewidth=0.9,
                   linestyle="--",
                   label=f"Tset={loop.settling_ms:.1f} ms")
    ax.set_xlabel("Time (ms)", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)


def _plot_root_locus(
    fig: plt.Figure,
    gs_loc,
    loop: LoopResult,
    title: str,
) -> None:
    """Root locus: sweep Kp with fixed Ki/Kp ratio, mark operating poles."""
    ax = fig.add_subplot(gs_loc)
    ax.set_title(title, fontsize=8)

    if loop._num_cl is None or loop.Kp <= 0:
        ax.text(0.5, 0.5, "No TF data", ha="center", va="center",
                transform=ax.transAxes)
        return

    tau = loop._tau_plant
    Rs = loop.Ki / loop.Kp * tau if loop.Kp > 0 else 1.0
    ratio = loop.Ki / loop.Kp

    Kp_sweep = np.linspace(0.01 * loop.Kp, 5.0 * loop.Kp, 200)
    real_parts, imag_parts = [], []
    for Kp_k in Kp_sweep:
        Ki_k = Kp_k * ratio
        num_cl_k = np.array([Kp_k, Ki_k])
        den_cl_k = np.array([Rs * tau, Rs + Kp_k, Ki_k])
        for r in np.roots(den_cl_k):
            real_parts.append(r.real)
            imag_parts.append(r.imag)

    cl_poles = np.roots(loop._den_cl)

    ax.scatter(real_parts, imag_parts, s=2, color="steelblue", alpha=0.3)
    ax.scatter(cl_poles.real, cl_poles.imag, s=50, color="crimson",
               zorder=5, marker="x", label="Operating poles")
    ax.axvline(0, color="gray", linewidth=0.7)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("Real", fontsize=7)
    ax.set_ylabel("Imag", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)


def _plot_waterfall_compact(
    fig: plt.Figure,
    gs_loc,
    plant: PMSMPlant,
    i_loop: LoopResult,
    w_loop: LoopResult,
    title: str,
    steps: int = 21,
) -> None:
    """PM vs Rs deviation for current and speed loops."""
    ax = fig.add_subplot(gs_loc)
    ax.set_title(title, fontsize=8)

    Rs_deltas = np.linspace(-0.3, 0.3, steps)
    Rs_nom = plant.Rs
    Ld_nom = plant.Ld
    Kp_i, Ki_i = i_loop.Kp, i_loop.Ki

    has_speed = w_loop is not None and w_loop._num_ol is not None
    Kp_w = w_loop.Kp if has_speed else 0.0
    Ki_w = w_loop.Ki if has_speed else 0.0
    Kt_eff = getattr(w_loop, "Kt_eff", 1.5 * plant.p * plant.psi_f)
    J_total = getattr(w_loop, "J_total", plant.J_total)
    tau_cl = getattr(w_loop, "tau_cl",
                     1.0 / (2.0 * math.pi * max(i_loop.BW_Hz, 1.0)))
    Kg = Kt_eff / J_total

    PM_curr = []
    PM_speed = []
    for dRs in Rs_deltas:
        Rs_v = Rs_nom * (1.0 + dRs)
        tau_v = Ld_nom / Rs_v
        num_i = np.array([Kp_i, Ki_i])
        den_i = np.array([Rs_v * tau_v, Rs_v, 0.0])
        pm_i, _ = _compute_margins(num_i, den_i)
        PM_curr.append(pm_i)

        if has_speed:
            num_w = Kg * np.array([Kp_w, Ki_w])
            den_w = np.array([tau_cl, 1.0, 0.0, 0.0])
            pm_w, _ = _compute_margins_speed(num_w, den_w)
            PM_speed.append(pm_w)

    Rs_pct = Rs_deltas * 100.0
    ax.plot(Rs_pct, PM_curr, color="steelblue", linewidth=1.2,
            label="Current loop")
    if PM_speed:
        ax.plot(Rs_pct, PM_speed, color="darkorange", linewidth=1.2,
                label="Speed loop")
    ax.axhline(_PM_MIN_DEG, color="crimson", linewidth=1.0,
               linestyle="--", label=f"PM={_PM_MIN_DEG:.0f} deg")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Rs deviation (%)", fontsize=7)
    ax.set_ylabel("PM (deg)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _write_csv(
    csv_path: str,
    i_loop_d: LoopResult,
    i_loop_q: LoopResult,
    w_loop: LoopResult,
) -> None:
    """
    Write gains summary CSV.

    Columns: loop, method, Kp, Ki, BW_Hz, PM_deg, GM_dB,
             settling_ms, overshoot_pct, warnings
    """
    rows = [
        ("current_d", i_loop_d),
        ("current_q", i_loop_q),
        ("speed",     w_loop),
    ]
    fieldnames = [
        "loop", "method", "Kp", "Ki", "BW_Hz", "PM_deg", "GM_dB",
        "settling_ms", "overshoot_pct", "warnings",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for loop_name, r in rows:
            gm = f"{r.GM_dB:.2f}" if r.GM_dB < 900 else "inf"
            writer.writerow({
                "loop": loop_name,
                "method": r.method,
                "Kp": f"{r.Kp:.6f}",
                "Ki": f"{r.Ki:.6f}",
                "BW_Hz": f"{r.BW_Hz:.2f}",
                "PM_deg": f"{r.PM_deg:.2f}",
                "GM_dB": gm,
                "settling_ms": f"{r.settling_ms:.3f}",
                "overshoot_pct": f"{r.overshoot_pct:.2f}",
                "warnings": "; ".join(r.warnings),
            })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Convert a name to a safe filename fragment."""
    return (
        str(name)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("-", "_")
    )
