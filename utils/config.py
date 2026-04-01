"""
utils/config.py — YAML config loader with validation and SI unit conversion.

All motor and load parameters are returned in SI base units:
  - resistance: Ω
  - inductance: H
  - flux linkage: Wb
  - inertia: kg·m²
  - friction: N·m·s/rad
  - torque: N·m
  - speed: rad/s  (converted from rpm at load time)
  - power: W
  - current: A

Usage:
    motor_cfg, load_cfg = load_config("config/motor_delta_ecma_c21010.yaml",
                                      "config/load_fan.yaml")
"""

import math
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Required keys — loader raises KeyError if any are missing
# ---------------------------------------------------------------------------

_MOTOR_REQUIRED_KEYS = {
    "motor_type": str,
    "name": str,
    "rated": {
        "power_W": float,
        "torque_Nm": float,
        "speed_rpm": float,
        "current_A": float,
    },
    "electrical": {
        "Rs_ohm": float,
        "Ld_H": float,
        "Lq_H": float,
        "psi_f_Wb": float,
        "pole_pairs": int,
    },
    "mechanical": {
        "J_kgm2": float,
        "B_Nms_rad": float,
    },
}

_LOAD_REQUIRED_KEYS = {
    "load_type": str,
    "name": str,
    "J_load_kgm2": float,
    "B_load_Nms_rad": float,
    "k_fan": float,
    "TL_Nm": float,
    "position_loop_active": bool,
}

_VALID_MOTOR_TYPES = {"SPMSM", "IPMSM"}
_VALID_LOAD_TYPES = {"fan", "constant_torque", "position_servo"}


def _check_keys(data: dict, schema: dict, path: str = "") -> None:
    """Recursively verify required keys exist in data."""
    for key, val_type in schema.items():
        full_key = f"{path}.{key}" if path else key
        if key not in data:
            raise KeyError(f"Missing required config key: '{full_key}'")
        if isinstance(val_type, dict):
            if not isinstance(data[key], dict):
                raise TypeError(f"Expected mapping for '{full_key}', got {type(data[key])}")
            _check_keys(data[key], val_type, full_key)


def _validate_motor(cfg: dict) -> None:
    """Validate motor config values for physical consistency."""
    motor_type = cfg["motor_type"]
    if motor_type not in _VALID_MOTOR_TYPES:
        raise ValueError(
            f"motor_type must be one of {_VALID_MOTOR_TYPES}, got '{motor_type}'"
        )

    elec = cfg["electrical"]
    Rs = elec["Rs_ohm"]
    Ld = elec["Ld_H"]
    Lq = elec["Lq_H"]
    psi_f = elec["psi_f_Wb"]
    p = elec["pole_pairs"]

    if Rs <= 0:
        raise ValueError(f"Rs_ohm must be > 0, got {Rs}")
    if Ld <= 0:
        raise ValueError(f"Ld_H must be > 0, got {Ld}")
    if Lq <= 0:
        raise ValueError(f"Lq_H must be > 0, got {Lq}")
    if psi_f <= 0:
        raise ValueError(f"psi_f_Wb must be > 0, got {psi_f}")
    if p < 1:
        raise ValueError(f"pole_pairs must be >= 1, got {p}")

    if motor_type == "SPMSM" and not math.isclose(Ld, Lq, rel_tol=0.05):
        raise ValueError(
            f"SPMSM requires Ld ≈ Lq, but Ld={Ld*1e3:.3f} mH, Lq={Lq*1e3:.3f} mH "
            f"(diff > 5%)"
        )

    if motor_type == "IPMSM" and Lq < Ld:
        raise ValueError(
            f"IPMSM requires Lq >= Ld, but Ld={Ld*1e3:.3f} mH > Lq={Lq*1e3:.3f} mH"
        )

    mech = cfg["mechanical"]
    if mech["J_kgm2"] <= 0:
        raise ValueError(f"J_kgm2 must be > 0, got {mech['J_kgm2']}")
    if mech["B_Nms_rad"] < 0:
        raise ValueError(f"B_Nms_rad must be >= 0, got {mech['B_Nms_rad']}")

    rated = cfg["rated"]
    if rated["speed_rpm"] <= 0:
        raise ValueError(f"rated speed_rpm must be > 0, got {rated['speed_rpm']}")


def _validate_load(cfg: dict) -> None:
    """Validate load config values."""
    load_type = cfg["load_type"]
    if load_type not in _VALID_LOAD_TYPES:
        raise ValueError(
            f"load_type must be one of {_VALID_LOAD_TYPES}, got '{load_type}'"
        )

    if cfg["J_load_kgm2"] < 0:
        raise ValueError(f"J_load_kgm2 must be >= 0, got {cfg['J_load_kgm2']}")
    if cfg["B_load_Nms_rad"] < 0:
        raise ValueError(f"B_load_Nms_rad must be >= 0, got {cfg['B_load_Nms_rad']}")

    if load_type == "fan" and cfg["k_fan"] <= 0:
        raise ValueError(f"Fan load requires k_fan > 0, got {cfg['k_fan']}")

    if load_type == "constant_torque" and cfg["TL_Nm"] < 0:
        raise ValueError(
            f"constant_torque load requires TL_Nm >= 0, got {cfg['TL_Nm']}"
        )


def _to_si_motor(cfg: dict) -> dict:
    """
    Convert motor config values to SI units in-place.
    Input YAML already uses SI (H, Ω, Wb, kg·m²) — this function adds derived fields.
    """
    elec = cfg["electrical"]
    rated = cfg["rated"]
    motor_type = cfg["motor_type"]

    # Enforce Ld = Lq for SPMSM (average if marginally different)
    if motor_type == "SPMSM":
        L_avg = (elec["Ld_H"] + elec["Lq_H"]) / 2.0
        elec["Ld_H"] = L_avg
        elec["Lq_H"] = L_avg

    # Derived electrical constants
    elec["tau_d_s"] = elec["Ld_H"] / elec["Rs_ohm"]
    elec["tau_q_s"] = elec["Lq_H"] / elec["Rs_ohm"]

    # Rated speed in rad/s
    rated["speed_rad_s"] = rated["speed_rpm"] * math.pi / 30.0

    # Saliency ratio
    elec["saliency_ratio"] = elec["Lq_H"] / elec["Ld_H"]

    return cfg


def _to_si_load(cfg: dict) -> dict:
    """Load config is already in SI. Add any derived fields."""
    return cfg


def load_config(motor_path: str, load_path: str) -> tuple[dict, dict]:
    """
    Load, validate, and return motor and load configs from YAML files.

    Parameters
    ----------
    motor_path : str or Path
        Path to motor YAML config (e.g., "config/motor_delta_ecma_c21010.yaml").
    load_path : str or Path
        Path to load YAML config (e.g., "config/load_fan.yaml").

    Returns
    -------
    motor_cfg : dict
        Validated motor parameters in SI units with derived fields added.
    load_cfg : dict
        Validated load parameters in SI units.

    Raises
    ------
    FileNotFoundError
        If either config file does not exist.
    KeyError
        If a required config key is missing.
    ValueError
        If a parameter value fails physical consistency checks.
    """
    motor_path = Path(motor_path)
    load_path = Path(load_path)

    if not motor_path.exists():
        raise FileNotFoundError(f"Motor config not found: {motor_path}")
    if not load_path.exists():
        raise FileNotFoundError(f"Load config not found: {load_path}")

    with motor_path.open("r") as f:
        motor_cfg = yaml.safe_load(f)
    with load_path.open("r") as f:
        load_cfg = yaml.safe_load(f)

    # Validate structure
    _check_keys(motor_cfg, _MOTOR_REQUIRED_KEYS)
    _check_keys(load_cfg, _LOAD_REQUIRED_KEYS)

    # Validate physical values
    _validate_motor(motor_cfg)
    _validate_load(load_cfg)

    # Add derived SI fields
    _to_si_motor(motor_cfg)
    _to_si_load(load_cfg)

    return motor_cfg, load_cfg
