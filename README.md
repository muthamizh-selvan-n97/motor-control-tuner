# motor-control-tuner

A practitioner's toolkit for PMSM Control Loop Tuning.

Given a motor datasheet, this project derives plant parameters, simulates parameter
identification, designs nested PI loops using four methods, analyses discretization
effects, and validates robustness — all reproducible from public data only.

---

## Features

- **Two reference motors** — SPMSM and IPMSM, from public datasheets
- **Plant model** — d-q frame state-space, MTPA, field weakening, discretization
- **Parameter identification** — simulated bench tests (Rs, Ld/Lq, psi_f) with noise
- **Current loop tuning** — pole-zero cancellation, frequency domain, root locus, Ziegler-Nichols
- **Speed loop tuning** — same four methods + anti-windup, bandwidth hierarchy check
- **Position loop** — P/PD + velocity feedforward
- **Discretization analysis** — Tustin vs ZOH, 1.5Ts PWM delay, Q15 quantization
- **Robustness analysis** — parameter sweep, sensitivity functions, margin waterfall
- **Dashboard** — 3×2 matplotlib figure + CSV export

---

## Motor parameters

### Motor 1: Delta ECMA-C21010 (SPMSM)

| Parameter | Symbol | Value | Unit |
|---|---|---|---|
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
**Derived values:** pole_pairs (from Kt/KE ratio), ψ_f (from KE constant), τ_e verified against datasheet (9.05 ms vs 9.30 ms datasheet)

---

### Motor 2: Magnetic S.r.l. BLQ-40 (IPMSM)

| Parameter | Symbol | Value | Unit |
|---|---|---|---|
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


---

## Repository structure

```
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
│   ├── param_id.py        # simulated bench tests + manual override
│   ├── current_loop.py    # 4 tuning methods + verification suite
│   ├── speed_loop.py      # 4 methods + anti-windup + BW hierarchy
│   ├── position_loop.py   # P/PD + velocity feedforward
│   ├── discretization.py  # Tustin vs ZOH, 1.5Ts delay, Q15
│   ├── robustness.py      # parameter sweep + sensitivity + waterfall
│   └── dashboard.py       # 3×2 matplotlib figure + CSV export
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
│   └── test_discretization.py
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

# Run all tests
.venv/Scripts/python -m pytest -v

# Launch notebook
.venv/Scripts/jupyter lab notebook/control_tuner.ipynb
```

---

## Load configurations

| File | Type | TL(ω) |
|---|---|---|
| `load_fan.yaml` | Fan (quadratic) | k·ω² |
| `load_const_torque.yaml` | Conveyor/compressor | constant |
| `load_position_servo.yaml` | Servo inertia | 0 |

Each load has `J_load_kgm2` and `B_load_Nms_rad` fields. The plant combines motor and load:

```
J_total = J_motor + J_load
B_total = B_motor + B_load
```

---

## Key equations

### PMSM voltage equations (d-q frame, motor convention)

```
vd = Rs·id + Ld·d(id)/dt − ωe·Lq·iq
vq = Rs·iq + Lq·d(iq)/dt + ωe·(Ld·id + ψf)
Te = (3/2)·p·(ψf·iq + (Ld − Lq)·id·iq)
```

### Pole-zero cancellation (current loop)

```
Plant:   Gd(s) = (1/Rs) / (τd·s + 1),   τd = Ld/Rs
PI zero: Ki/Kp = Rs/Ld   (cancels plant pole)
Gains:   Kp = Ld/τcl,   Ki = Rs/τcl
```

### PWM delay phase loss

```
Phase loss = 1.5 · Ts · ωc · (180/π)  degrees
```

---

## Running the full pipeline

```bash
.venv/Scripts/python run_all.py
```

Runs both motors × both loads, saves all plots and a gains summary CSV to `outputs/`.

---

## Requirements

- Python 3.10+
- numpy, scipy, matplotlib, pyyaml, jupyter, control, pytest

See `requirements.txt` for pinned versions.
