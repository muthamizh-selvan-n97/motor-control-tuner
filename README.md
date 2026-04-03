# motor-control-tuner

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muthamizh-selvan-n97/motor-control-tuner/blob/master/notebook/control_tuner.ipynb)

A practitioner's toolkit for PMSM control loop tuning.

Given a motor datasheet, this project derives plant parameters, simulates full
system identification, designs nested PI loops using four methods, analyses
discretization effects, and validates robustness — all reproducible from
public data only.

---

## Features

- **Two reference motors** — SPMSM (Delta ECMA-C21010) and IPMSM (Magnetic BLQ-40), from public datasheets
- **Plant model** — d-q frame state-space, MTPA, field weakening, discretization
- **Electrical identification** — simulated bench tests: Rs (DC step), Ld/Lq (AC injection), psi_f (BEMF fit)
- **Mechanical identification** — pole pairs, KE_SI, J_total/J_load (acceleration ramp), B_total/k_fan (torque sweep)
- **Current loop tuning** — pole-zero cancellation, frequency domain, root locus, Ziegler-Nichols
- **Speed loop tuning** — same four methods + back-calculation anti-windup + BW hierarchy check
- **Position loop** — P/PD controllers + velocity feedforward, BW/10 hierarchy enforced
- **Discretization analysis** — Tustin vs ZOH, 1.5Ts PWM delay (Padé), Q15 fixed-point quantization
- **Robustness analysis** — Rs/L parameter sweep, sensitivity functions, PM/GM waterfall
- **Dashboard** — 3×2 matplotlib figure + CSV export of all gains and margins

---

## Motor parameters

### Motor 1: Delta ECMA-C21010 (SPMSM)

| Parameter | Symbol | Value | Unit |
| --- | --- | --- | --- |
| Rated power | P | 1000 | W |
| Rated torque | T_rated | 3.18 | N·m |
| Rated speed | n_rated | 3000 | rpm |
| Rated current | I_rated | 7.30 | A (phase, peak) |
| Resistance | Rs | 0.20 | Ω |
| Inductance | Ld = Lq | 1.81 | mH |
| Flux linkage | ψ_f | 0.0436 | Wb |
| Pole pairs | p | 4 | — |
| Rotor inertia | J | 2.65×10⁻⁴ | kg·m² |

**Source:** Delta Electronics ASDA-B2 Series User Manual, Chapter 11  
**Parameters extracted from:** Table "Low Inertia Servo Motor", column C210, row 1000W  
**Derived values:** pole_pairs (from Kt/KE ratio), ψ_f (from KE constant), τ_e verified against datasheet (9.05 ms vs 9.30 ms)  
**Download:** [ECMAServo-datasheet.pdf](https://kwoco-plc.com/wp-content/uploads/2024/05/ECMAServo-datasheet.pdf)

---

### Motor 2: Magnetic S.r.l. BLQ-40 (IPMSM)

| Parameter | Symbol | Value | Unit |
| --- | --- | --- | --- |
| Rated power | P | 2000 | W |
| Rated torque | T_rated | 6.37 | N·m |
| Rated speed | n_rated | 3000 | rpm |
| Rated current | I_rated | 5.6 | A (rms) |
| Resistance | Rs | 0.47 | Ω |
| d-axis inductance | Ld | 5.99 | mH |
| q-axis inductance | Lq | 9.80 | mH |
| Flux linkage | ψ_f | 0.255 | Wb |
| Pole pairs | p | 3 | — |
| Saliency ratio | Lq/Ld | 1.64 | — |
| Rotor inertia | J | 8.41×10⁻⁴ | kg·m² |

**Source:** Caruso et al., "Characterization of the parameters of interior permanent magnet
synchronous motors for a loss model algorithm," *Measurement*, Elsevier, 2017.  
**DOI:** [10.1016/j.measurement.2017.04.041](https://doi.org/10.1016/j.measurement.2017.04.041)  
**Parameters extracted from:** Table 1, "Rated values and parameters of the IPMSM under test"

---

## Repository structure

```text
motor-control-tuner/
├── requirements.txt
├── run_all.py
│
├── config/
│   ├── motor_delta_ecma_c21010.yaml   # SPMSM reference motor
│   ├── motor_magnetic_blq40.yaml      # IPMSM reference motor
│   ├── load_fan.yaml                  # TL ∝ ω² (quadratic)
│   ├── load_const_torque.yaml         # TL = constant (conveyor)
│   └── load_position_servo.yaml       # inertia load, position loop
│
├── modules/
│   ├── plant.py           # PMSM d-q state-space, MTPA, field weakening
│   ├── param_id.py        # electrical + mechanical identification
│   ├── current_loop.py    # 4 tuning methods + verification suite
│   ├── speed_loop.py      # 4 methods + anti-windup + BW hierarchy
│   ├── position_loop.py   # P/PD + velocity feedforward
│   ├── discretization.py  # Tustin vs ZOH, 1.5Ts delay, Q15
│   ├── robustness.py      # parameter sweep + sensitivity + waterfall
│   └── dashboard.py       # 3×2 matplotlib figure + CSV export
│
├── docs/
│   ├── theory_pmsm_model.md
│   ├── theory_current_loop.md
│   ├── theory_speed_loop.md
│   ├── theory_discretization.md
│   └── theory_position_loop.md
│
├── notebook/
│   └── control_tuner.ipynb
│
├── utils/
│   └── config.py          # YAML loader with SI conversion + validation
│
├── test/
│   ├── test_plant.py
│   ├── test_param_id.py
│   ├── test_current_loop.py
│   ├── test_speed_loop.py
│   ├── test_discretization.py
│   ├── test_robustness.py
│   ├── test_dashboard.py
│   └── test_position_loop.py
│
└── outputs/               # generated plots and CSVs (not tracked in git)
```

---

## Quick start

```bash
# Clone and set up
git clone <repo-url>
cd motor-control-tuner
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# or: .venv/bin/pip install -r requirements.txt  # Linux/macOS

# Run all tests (325 tests)
.venv/Scripts/python -m pytest -v

# Run full pipeline (both motors × all loads → outputs/)
.venv/Scripts/python run_all.py

# Launch notebook
.venv/Scripts/jupyter lab notebook/control_tuner.ipynb
```

---

## Notebook workflow

The notebook (`notebook/control_tuner.ipynb`) walks through the full pipeline:

| Cell | Stage | Output |
| --- | --- | --- |
| 1 | Motor + load config | — |
| 2 | Plant model + MTPA | Transfer functions |
| 3 | Electrical identification | Rs, Ld, Lq, psi_f |
| 4 | Mechanical identification | p, KE_SI, J_total, B_total, k_fan |
| 5–7 | Current loop (4 methods) | Kp, Ki, Bode, step plots |
| 8–9 | Discretization | Tustin/ZOH table, delay Bode |
| 10–13 | Speed loop (4 methods) | Kp, Ki, Kb, anti-windup |
| 14–16 | Robustness | Sweep, sensitivity, waterfall |
| 17 | Dashboard + CSV export | `outputs/dashboard_*.png` |
| 18–19 | Position loop (servo) | Kp, Kd, Kff_v |

To switch motor or load, edit the two `load_config()` lines in Cell 1.

---

## Load configurations

| File | Type | TL(ω) |
| --- | --- | --- |
| `load_fan.yaml` | Fan (quadratic) | k·ω² |
| `load_const_torque.yaml` | Conveyor/compressor | constant |
| `load_position_servo.yaml` | Servo inertia | 0 |

```text
J_total = J_motor + J_load
B_total = B_motor + B_load
```

---

## Identification methods

### Electrical (Tests 1–3)

| Test | Parameter | Method |
| --- | --- | --- |
| 1 | Rs | DC lockout step — measure Id steady state |
| 2 | Ld, Lq | AC standstill injection at 100 Hz, two rotor positions |
| 3 | psi_f | Open-circuit BEMF at 1000/2000/3000 rpm, OLS fit |

### Mechanical (Tests 4–7, require load config)

| Test | Parameter | Method |
| --- | --- | --- |
| 4 | p (pole pairs) | Electrical frequency count at 1000 rpm — integer rounding |
| 5 | KE_SI, psi_f_ke | OLS fit of BEMF slope vs ω — KE_SI = psi_f·p |
| 6 | J_total, J_load | No-load acceleration ramp — linear fit over 500 samples |
| 7 | B_total, k_fan / TL_const | Steady-state torque sweep at 10 speeds — lstsq per load type |

To use real bench measurements instead of simulation:

```python
params      = pid.override(Rs=0.21, Ld=1.85e-3, Lq=1.85e-3, psi_f=0.044)
mech_params = pid.override(p=4, J_total=1.06e-3, B_total=0.001, k_fan=1.12e-5)
```

---

## Key equations

### PMSM voltage equations (d-q frame)

```text
vd = Rs·id + Ld·d(id)/dt − ωe·Lq·iq
vq = Rs·iq + Lq·d(iq)/dt + ωe·(Ld·id + ψf)
Te = (3/2)·p·(ψf·iq + (Ld − Lq)·id·iq)
```

### Pole-zero cancellation (current loop)

```text
Plant:   Gd(s) = (1/Rs) / (τd·s + 1),   τd = Ld/Rs
PI zero: Ki/Kp = Rs/Ld   (cancels plant pole)
Gains:   Kp = Ld/τcl,   Ki = Rs/τcl
```

### Bandwidth hierarchy

```text
BW_pos <= BW_speed / 10 <= BW_current / 100
```

### Position loop

```text
Plant:   G_pos(s) = 1 / (s·(τw·s + 1)),   τw = 1/(2π·BW_speed)
P:       Kp_pos = ωc·√(1 + (τw·ωc)²)
PD:      Kd_pos = 2·ζ / ωc   (crossover pinned at ωc)
e_ss (ramp, Kff_v=0) = 1 / Kp_pos
```

### PWM delay phase loss

```text
Phase loss = 1.5 · Ts · ωc · (180/π)  degrees
At ωc = 2π·500 Hz, Ts = 50 µs:  loss ≈ 13.5°
```

---

## How to use with your own motor and load

### Step 1 — Add your motor config

Copy `config/motor_delta_ecma_c21010.yaml` and fill in your datasheet values:

```yaml
motor_type: SPMSM          # or IPMSM if Ld != Lq
name: "My Motor"
rated:
  power_W: 750
  torque_Nm: 2.4
  speed_rpm: 3000
  current_A: 5.0
electrical:
  Rs_ohm: 0.35
  Ld_H: 2.5e-3
  Lq_H: 2.5e-3            # set Lq = Ld for SPMSM
  psi_f_Wb: 0.055          # derived: KE_mVrpm * 1e-3 * 30/pi / pole_pairs
  pole_pairs: 3
mechanical:
  J_kgm2: 1.5e-4
  B_Nms_rad: 0.001         # estimate if not on datasheet
```

For IPMSM set `Lq_H > Ld_H`. The saliency ratio enables MTPA and field weakening.

### Step 2 — Add your load config

Copy the closest load template:

```yaml
# Fan / pump
load_type: fan
J_load_kgm2: 4.5e-4
B_load_Nms_rad: 0.0
k_fan: 8.0e-6              # TL = k_fan * omega^2; fit from measured speed/torque point

# Conveyor / compressor
load_type: constant_torque
J_load_kgm2: 3.0e-4
B_load_Nms_rad: 0.0
TL_Nm: 1.5

# Servo / positioning
load_type: position_servo
J_load_kgm2: 8.0e-4
B_load_Nms_rad: 0.0
TL_Nm: 0.0
```

### Step 3 — Select your config in the notebook

In Cell 1 of `notebook/control_tuner.ipynb`, change the two paths:

```python
motor_cfg, load_cfg = load_config(
    '../config/motor_my_motor.yaml',
    '../config/load_my_load.yaml'
)
```

### Step 4 — Run identification

**Simulation (no bench hardware):** Cells 3 and 4 simulate all seven tests automatically.

**Real bench measurements:** Replace simulate calls with override:

```python
# Electrical (from bench)
params = pid.override(
    Rs=0.36,        # DC step test
    Ld=2.48e-3,     # AC injection, d-axis position
    Lq=2.52e-3,     # AC injection, q-axis position
    psi_f=0.054,    # BEMF at known speed
)

# Mechanical (from bench)
mech_params = pid.override(
    p=3,             # BEMF zero-crossing count
    J_total=6.5e-4,  # no-load acceleration test
    B_total=0.0012,  # steady-state torque sweep
    k_fan=8.1e-6,    # torque sweep (fan load only)
)
```

### Step 5 — Tune the loops

**Current loop** — the main knob is `tau_cl_s` (desired closed-loop time constant).
Typical starting point: `tau_cl_s = 5 * Ts_PWM` to `10 * Ts_PWM`.

```python
# At 10 kHz PWM (Ts = 100 µs), try tau_cl = 0.5–1 ms
i_loop = tuner_i.tune('pole_zero', axis='d', tau_cl_s=1e-3)
i_loop.summary()   # check PM_deg >= 45, GM_dB >= 6
```

Decrease `tau_cl_s` to increase bandwidth. Stop when `PM_deg` drops below 45°
or warnings appear in `i_loop.warnings`.

**Speed loop** — keep `BW_speed <= BW_current / 10`:

```python
w_loop = tuner_w.tune('pole_zero', tau_cl_s=50e-3)
w_loop.summary()
```

**Position loop** — defaults to maximum stable BW (`BW_speed / 10`):

```python
p_loop = tuner_pos.tune('PD', zeta=0.7, Kff_v=1.0)   # Kff_v=1 for zero ramp error
p_loop.summary()
```

### Step 6 — Check discretization

Set your actual PWM period and inspect the firmware table:

```python
disc = Discretizer(i_loop, Ts_s=100e-6)   # 10 kHz
disc.pwm_delay_analysis()                  # check phase loss at crossover
print(disc.firmware_table())               # copy Kp/Ki into your firmware
```

Rule of thumb: achievable current loop BW ≈ 1 / (6 × Ts).
At 10 kHz that is ≈ 1.67 kHz maximum.

### Step 7 — Validate robustness

```python
rob = RobustnessAnalyser(plant, i_loop, w_loop)
rob.parameter_sweep()    # PM/GM vs Rs ±30%, L ±20%
rob.margin_waterfall()   # worst-case margin across temperature range
```

Aim for PM > 45° and GM > 6 dB across the full sweep before deploying gains.

---

## Requirements

- Python 3.10+
- numpy, scipy, matplotlib, pyyaml, jupyter, control, pytest

See `requirements.txt` for pinned versions.
