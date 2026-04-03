"""
run_all.py — Run full pipeline for both motors x all loads. Saves all outputs.

Usage:
    python run_all.py
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils.config import load_config
from modules.plant import PMSMPlant
from modules.param_id import ParameterIdentifier
from modules.current_loop import CurrentLoopTuner
from modules.speed_loop import SpeedLoopTuner
from modules.position_loop import PositionLoopTuner
from modules.discretization import Discretizer
from modules.robustness import RobustnessAnalyser
from modules.dashboard import generate_dashboard

MOTORS = [
    "config/motor_delta_ecma_c21010.yaml",
    "config/motor_magnetic_blq40.yaml",
]
LOADS = [
    "config/load_fan.yaml",
    "config/load_const_torque.yaml",
]
SERVO_LOAD = "config/load_position_servo.yaml"

OUTPUT_DIR = "outputs"
TAU_CL_CURRENT = 1e-3   # current loop closed-loop time constant (s)
TAU_CL_SPEED = 50e-3    # speed loop closed-loop time constant (s)
TS_PWM = 50e-6           # PWM period (s) — 20 kHz


def run_one(motor_path: str, load_path: str) -> None:
    motor_cfg, load_cfg = load_config(motor_path, load_path)
    plant = PMSMPlant(motor_cfg, load_cfg)

    print(f"\n{'='*60}")
    print(f"Motor : {motor_cfg['name']}  [{motor_cfg['motor_type']}]")
    print(f"Load  : {load_cfg['name']}  [{load_cfg['load_type']}]")
    print(f"{'='*60}")

    # --- Parameter identification ---
    params = ParameterIdentifier(motor_cfg).simulate()

    # --- Current loop ---
    tuner_i = CurrentLoopTuner(plant, params)
    i_loop_d = tuner_i.tune("pole_zero", axis="d", tau_cl_s=TAU_CL_CURRENT)
    i_loop_q = tuner_i.tune("pole_zero", axis="q", tau_cl_s=TAU_CL_CURRENT)
    print(f"Current loop d: Kp={i_loop_d.Kp:.4f}  Ki={i_loop_d.Ki:.2f}  "
          f"BW={i_loop_d.BW_Hz:.1f} Hz  PM={i_loop_d.PM_deg:.1f} deg")
    print(f"Current loop q: Kp={i_loop_q.Kp:.4f}  Ki={i_loop_q.Ki:.2f}  "
          f"BW={i_loop_q.BW_Hz:.1f} Hz  PM={i_loop_q.PM_deg:.1f} deg")

    # --- Discretization ---
    disc = Discretizer(i_loop_d, Ts_s=TS_PWM)
    disc.firmware_table()

    # --- Speed loop ---
    tuner_w = SpeedLoopTuner(plant, params, i_loop_d)
    w_loop = tuner_w.tune("pole_zero", tau_cl_s=TAU_CL_SPEED)
    print(f"Speed loop    : Kp={w_loop.Kp:.4f}  Ki={w_loop.Ki:.4f}  "
          f"BW={w_loop.BW_Hz:.1f} Hz  PM={w_loop.PM_deg:.1f} deg  Kb={w_loop.Kb:.4f}")
    for w in w_loop.warnings:
        print(f"  {w}")

    # --- Robustness ---
    rob = RobustnessAnalyser(plant, i_loop_d, w_loop)
    rob.sensitivity(
        axis="d",
        save_path=os.path.join(
            OUTPUT_DIR,
            f"sensitivity_{_safe(motor_cfg['name'])}_{_safe(load_cfg['name'])}.png",
        ),
    )
    rob.margin_waterfall(
        steps=21,
        axis="d",
        save_path=os.path.join(
            OUTPUT_DIR,
            f"waterfall_{_safe(motor_cfg['name'])}_{_safe(load_cfg['name'])}.png",
        ),
    )

    # --- Dashboard ---
    generate_dashboard(
        params,
        i_loop_d,
        i_loop_q,
        w_loop,
        plant=plant,
        save=True,
        output_dir=OUTPUT_DIR,
    )

    print(f"Done: {motor_cfg['name']} + {load_cfg['name']}")


def run_servo(motor_path: str) -> None:
    """Run current → speed → position loop pipeline for the servo load."""
    motor_cfg, load_cfg = load_config(motor_path, SERVO_LOAD)
    plant = PMSMPlant(motor_cfg, load_cfg)

    print(f"\n{'='*60}")
    print(f"Motor : {motor_cfg['name']}  [{motor_cfg['motor_type']}]")
    print(f"Load  : {load_cfg['name']}  [{load_cfg['load_type']}]")
    print(f"{'='*60}")

    params = ParameterIdentifier(motor_cfg).simulate()

    # Current loop
    tuner_i = CurrentLoopTuner(plant, params)
    i_loop_d = tuner_i.tune("pole_zero", axis="d", tau_cl_s=TAU_CL_CURRENT)
    i_loop_q = tuner_i.tune("pole_zero", axis="q", tau_cl_s=TAU_CL_CURRENT)
    print(f"Current loop d: Kp={i_loop_d.Kp:.4f}  Ki={i_loop_d.Ki:.2f}  "
          f"BW={i_loop_d.BW_Hz:.1f} Hz  PM={i_loop_d.PM_deg:.1f} deg")

    # Speed loop
    tuner_w = SpeedLoopTuner(plant, params, i_loop_d)
    w_loop = tuner_w.tune("pole_zero", tau_cl_s=TAU_CL_SPEED)
    print(f"Speed loop    : Kp={w_loop.Kp:.4f}  Ki={w_loop.Ki:.4f}  "
          f"BW={w_loop.BW_Hz:.1f} Hz  PM={w_loop.PM_deg:.1f} deg")

    # Position loop — P and PD
    tuner_p = PositionLoopTuner(plant, w_loop)
    p_loop_P  = tuner_p.tune("P",  Kff_v=1.0)
    p_loop_PD = tuner_p.tune("PD", zeta=0.7, Kff_v=1.0)

    print(f"Position P    : Kp={p_loop_P.Kp:.4f}  "
          f"BW={p_loop_P.BW_Hz:.2f} Hz  PM={p_loop_P.PM_deg:.1f} deg  "
          f"settle={p_loop_P.settling_ms:.1f} ms")
    print(f"Position PD   : Kp={p_loop_PD.Kp:.4f}  Kd={p_loop_PD.Kd:.4f}  "
          f"BW={p_loop_PD.BW_Hz:.2f} Hz  PM={p_loop_PD.PM_deg:.1f} deg  "
          f"settle={p_loop_PD.settling_ms:.1f} ms")
    for w in p_loop_PD.warnings:
        print(f"  {w}")

    # Save position loop plots
    p_loop_PD.plot_bode()
    p_loop_PD.plot_step()

    # Dashboard (position servo, use PD result as reference)
    generate_dashboard(
        params,
        i_loop_d,
        i_loop_q,
        w_loop,
        plant=plant,
        save=True,
        output_dir=OUTPUT_DIR,
    )

    print(f"Done: {motor_cfg['name']} + {load_cfg['name']}")


def _safe(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fan and constant-torque loads — full pipeline
    for motor_path, load_path in itertools.product(MOTORS, LOADS):
        run_one(motor_path, load_path)

    # Servo load — adds position loop
    for motor_path in MOTORS:
        run_servo(motor_path)

    print("\nAll combinations complete.")
