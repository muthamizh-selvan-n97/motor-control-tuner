# Position Loop Tuning Theory

## 1. Control Structure

The position loop is the outermost loop in the nested control hierarchy:

```
Position ref → [C_pos] → Speed ref → [C_speed] → Current ref → [C_current] → Motor
```

**Bandwidth hierarchy (mandatory):**

```
BW_pos ≤ BW_speed / 10 ≤ BW_current / 100
```

This separation ensures each inner loop is fast enough to appear as near-unity gain to the outer loop during design.

### Plant

The closed speed loop is approximated as a first-order lag:

```
G_speed_cl(s) ≈ 1 / (τw·s + 1)      τw = 1 / (2π·BW_speed)
```

Adding the integrator that converts speed to position:

```
G_pos(s) = G_speed_cl(s) / s = 1 / (s·(τw·s + 1))
```

When BW_pos << BW_speed, the lag τw is negligible and the plant simplifies to a pure integrator `1/s`.

---

## 2. Method 1 — P Controller

**Controller:**

```
C(s) = Kp_pos
```

**Open-loop:**

```
L(s) = Kp_pos / (s·(τw·s + 1))
```

**Gain crossover condition** `|L(jωc)| = 1`:

```
Kp_pos / (ωc·√(1 + (τw·ωc)²)) = 1

=> Kp_pos = ωc·√(1 + (τw·ωc)²)
```

When τw << 1/ωc this simplifies to `Kp_pos ≈ ωc = 2π·BW_pos`.

**Phase margin:**

```
∠L(jωc) = −90° − arctan(τw·ωc)

PM = 90° − arctan(τw·ωc)
```

For BW_pos = BW_speed/10, τw·ωc = 0.1 → PM ≈ 84°.

**Closed-loop:**

```
T(s) = Kp_pos / (τw·s² + s + Kp_pos)
```

Dominant pole at s ≈ −Kp_pos (when τw is small).

---

## 3. Method 2 — PD Controller

**Controller:**

```
C(s) = Kp_pos·(1 + Kd·s)
```

The derivative term adds a zero at `s = −1/Kd`, providing phase lead at the crossover frequency.

**Derivative gain** (chosen to set damping ratio ζ at crossover):

```
Kd = 2·ζ / ωc        (default ζ = 0.7)
```

**Proportional gain** (crossover pinned at ωc):

Set `|L(jωc)| = 1`:

```
Kp·√(1 + (Kd·ωc)²) / (ωc·√(1 + (τw·ωc)²)) = 1

=> Kp = ωc·√(1 + (τw·ωc)²) / √(1 + (Kd·ωc)²)
```

With ζ = 0.7: `(Kd·ωc)² = (2·0.7)² = 1.96`, so `Kp_PD = Kp_P / √(1 + 4·ζ²)`.

**Closed-loop:**

```
T(s) = Kp·(Kd·s + 1) / (τw·s² + (1 + Kp·Kd)·s + Kp)
```

The added damping `(1 + Kp·Kd)` improves PM at the cost of a lower Kp, so settling is
slower than P in the time domain. The benefit is robustness: higher PM means better
tolerance to plant uncertainty.

---

## 4. Velocity Feedforward

Without feedforward, a ramp input `r(t) = v·t` produces a steady-state following error:

```
e_ss = v / Kp_pos      [position units per unit ramp rate]
```

This is the **velocity error constant** of a Type-1 loop (one integrator in the plant).

With velocity feedforward gain `Kff_v`:

```
e_ss = v·(1 − Kff_v) / Kp_pos
```

Setting `Kff_v = 1.0` gives zero steady-state following error for any ramp rate,
without changing the feedback stability margins.

**Implementation:** the feedforward signal bypasses the position controller and adds
directly to the speed reference:

```
ω_ref = C_pos·(θ_ref − θ) + Kff_v·dθ_ref/dt
```

---

## 5. Stability Margins

| Source of phase loss | Magnitude |
| --- | --- |
| Position integrator 1/s | −90° |
| Speed-loop lag τw | −arctan(τw·ωc) ≈ −6° at BW_pos = BW_speed/10 |
| **Total open-loop phase** | ≈ −96° |
| **P controller PM** | ≈ 84° |
| PD zero (ζ=0.7) | +arctan(Kd·ωc) = +arctan(2ζ) ≈ +54° |
| **PD controller PM** | ≈ 138° |

Both methods satisfy PM ≥ 45°. The high margins reflect the conservatively low BW_pos.

---

## 6. Relationship to Inner Loops

| Loop | Plant order | Controller | Typical BW |
| --- | --- | --- | --- |
| Current (d/q) | 1st order (R-L) | PI | 200–500 Hz |
| Speed | 2nd order (integrator + CL lag) | PI | 20–50 Hz |
| Position | 2nd order (integrator + speed lag) | P or PD | 0.3–5 Hz |

The position loop sees the closed speed loop as its plant. Errors in speed-loop tracking
appear as disturbances to the position loop. The BW/10 hierarchy ensures these
disturbances are attenuated before they reach the position reference.

---

## 7. Design Guidelines

| Parameter | Guideline |
| --- | --- |
| BW_pos | ≤ BW_speed / 10 |
| Default BW_pos | = BW_speed / 10 (maximum stable) |
| ζ (PD) | 0.5–0.8 (0.7 is a good default) |
| Kff_v | 1.0 for point-to-point servo; 0 if position reference is pre-filtered |
| PM (P) | ≈ 84° at BW_pos = BW_speed/10 |
| PM (PD, ζ=0.7) | ≈ 138° (overdamped, very robust) |

---

## 8. References

- Franklin, G.F., Powell, J.D. & Emami-Naeini, A., *Feedback Control of Dynamic Systems*, Pearson, 2015. Chapter 4 (steady-state errors) and Chapter 9 (cascade control).
- Mohan, N., *Advanced Electric Drives*, Wiley, 2014. Chapter 11 (position control).
- Ellis, G., *Control System Design Guide*, Elsevier, 2012. Chapter 7 (position loop design).
