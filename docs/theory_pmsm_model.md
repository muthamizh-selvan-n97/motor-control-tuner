# PMSM Plant Model Theory

## 1. Physical Description

A Permanent Magnet Synchronous Motor (PMSM) produces torque through the interaction of stator currents and a permanent-magnet rotor flux. Analysis is carried out in the **d-q rotating reference frame**, which is synchronous with the rotor. In this frame, all AC quantities become DC at steady state, making PI control straightforward.

Two motor variants are supported:

| Type | Characteristic | Ld vs Lq |
|---|---|---|
| SPMSM | Surface magnets, uniform air gap | Ld ≈ Lq |
| IPMSM | Interior magnets, non-uniform air gap | Lq > Ld (saliency) |

---

## 2. Voltage Equations (Motor Convention)

```
vd = Rs·id + Ld·d(id)/dt − ωe·Lq·iq          (d-axis)
vq = Rs·iq + Lq·d(iq)/dt + ωe·(Ld·id + ψf)   (q-axis)
```

| Symbol | Description | Unit |
|---|---|---|
| vd, vq | d-q axis stator voltages | V |
| id, iq | d-q axis stator currents | A |
| Rs | Stator phase resistance | Ω |
| Ld, Lq | d-q axis inductances | H |
| ωe | Electrical angular velocity | rad/s |
| ψf | Permanent magnet flux linkage | Wb |

The cross-coupling terms `−ωe·Lq·iq` and `+ωe·Ld·id` are treated as **measurable disturbances** and cancelled by feedforward in the controller. After cancellation the two axes are decoupled first-order systems.

---

## 3. Decoupled Per-Axis Plant

After feedforward decoupling:

```
G_d(s) = id(s)/vd(s) = (1/Rs) / (τd·s + 1)    τd = Ld/Rs
G_q(s) = iq(s)/vq(s) = (1/Rs) / (τq·s + 1)    τq = Lq/Rs
```

DC gain: `1/Rs` (A/V).  
Time constant: τ = L/Rs (electrical time constant).

For the SPMSM: τd = τq = τe = L/Rs.

---

## 4. Electromagnetic Torque

```
Te = (3/2)·p·[ψf·iq + (Ld − Lq)·id·iq]
```

- First term `ψf·iq`: **alignment torque** (present in both SPMSM and IPMSM).
- Second term `(Ld−Lq)·id·iq`: **reluctance torque** (zero for SPMSM since Ld = Lq).

---

## 5. Mechanical Equation

```
J_total · dωm/dt = Te − TL − B·ωm
ωe = p · ωm
```

| Symbol | Description |
|---|---|
| J_total | Total inertia (motor + load) |
| TL | Load torque |
| B | Total viscous friction |
| p | Pole pairs |
| ωm | Mechanical angular velocity |

---

## 6. Speed Plant

With the current loop closed (approximated as unity for ω << BW_current), the speed plant is an integrator:

```
G_speed(s) = ωm(s)/iq*(s) = Kt_eff / (J_total·s + B)
```

where `Kt_eff = (3/2)·p·ψf`.

For the full model including the current loop closed-loop lag (τcl = 1/(2π·BWcurrent)):

```
G_speed_full(s) = (Kt_eff/J_total) / (s·(τcl·s + 1))
```

---

## 7. MTPA — Maximum Torque Per Ampere (IPMSM only)

For a given stator current magnitude Is, MTPA minimises copper losses:

```
id_MTPA = (ψf − √(ψf² + 8·(Lq−Ld)²·Is²)) / (4·(Lq−Ld))
iq_MTPA = √(Is² − id_MTPA²)
β_MTPA  = arctan(−id_MTPA / iq_MTPA)   [current angle]
```

For SPMSM: `id* = 0` always (id=0 control gives MTPA since there is no reluctance torque).

---

## 8. Field Weakening (above base speed)

Above base speed the back-EMF exceeds the available voltage. id* is reduced below zero to weaken the airgap flux:

```
Vmax = Vdc / √3              (voltage limit, per-phase peak)
id_FW = (−ψf + √(Vmax²/ωe² − Lq²·iq²)) / Ld
```

id_FW ≤ 0, and |id_FW| is limited so that id² + iq² ≤ Is_max².

---

## 9. State-Space Representation

Current plant (2-axis, decoupled):

```
ẋ = A·x + B·u
y = C·x

x = [id, iq]ᵀ,  u = [vd, vq]ᵀ

A = [−Rs/Ld    0    ]    B = [1/Ld   0  ]    C = I₂
    [  0    −Rs/Lq  ]        [  0   1/Lq]
```

---

## 10. Load Models

| Load type | Torque equation | Notes |
|---|---|---|
| Fan | TL = k_fan · ωm² | Linearised at rated ω for loop design |
| Constant torque | TL = const | Conveyor, compressor |
| Position servo | TL ≈ 0 | Inertia dominated |

For the fan load, linearisation around ωrated gives an effective damping addition:

```
B_eff = B_total + 2·k_fan·ωrated
```

---

## References

- Mohan, N., *Advanced Electric Drives*, Wiley, 2014.
- Holmes, D.G. et al., *Pulse Width Modulation for Power Converters*, IEEE Press, 2003.
- Caruso et al., "Characterization of the parameters of interior permanent magnet synchronous motors," *Measurement*, Elsevier, 2017. DOI: 10.1016/j.measurement.2017.04.041
