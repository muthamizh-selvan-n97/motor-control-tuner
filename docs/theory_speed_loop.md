# Speed Loop Tuning Theory

## 1. Control Structure

The speed loop is the outer loop in the cascade architecture:

```
ωref → [PI speed] → iq* → [PI current] → vq* → [PWM] → Motor → ω
```

**Bandwidth hierarchy:** The speed loop bandwidth must be at least one decade below the current loop bandwidth to preserve the cascade assumption:

```
BWspeed ≤ BWcurrent / 10
```

---

## 2. Speed Plant

With the current loop closed and approximated as first-order lag:

```
G_speed(s) = (Kt_eff/J_total) / (s · (τcl·s + 1))

Kt_eff = (3/2)·p·ψf      [effective torque constant, N·m/A]
τcl    = 1/(2π·BWcurrent)  [current loop time constant]
J_total = J_motor + J_load
```

For frequencies ω << 1/τcl the lag is negligible and the plant simplifies to a pure integrator:

```
G_speed_approx(s) ≈ Kt_eff / (J_total·s)
```

---

## 3. Fan Load Linearisation

For a fan/pump load `TL = k_fan·ωm²`, linearised around the rated operating point ωr:

```
ΔTL = 2·k_fan·ωr·Δωm
```

This appears as an additional damping term:

```
B_eff = B_total + 2·k_fan·ωr
```

The linearised plant becomes:

```
G_speed(s) = (Kt_eff/J_total) / (s + B_eff/J_total)
```

For design purposes B_eff/J_total is small compared to the crossover frequency and is often neglected in gain calculation.

---

## 4. Method 1 — Pole-Zero Cancellation

For the speed plant (pure integrator approximation):

```
Kp = J_total / (Kt_eff · τw)
```

where τw is the desired speed loop closed-loop time constant.

For Ki, two cases:

- **B_eff > 0** (fan or load with viscous friction): Cancel the mechanical pole at s = −B_eff/J_total:
  ```
  Ki = Kp · B_eff / J_total
  ```
- **B_eff ≈ 0** (pure inertia): Place integral one decade below crossover:
  ```
  Ki = Kp / (10·τw)
  ```

---

## 5. Method 2 — Frequency Domain

At crossover ωc = 2π·BW_Hz, `|C(jωc)·G_speed(jωc)| = 1`.

Speed plant magnitude at ωc (with τcl lag):

```
|G_speed(jωc)| = (Kt_eff/J_total) / (ωc · √(1 + (τcl·ωc)²))
```

With Ki = Kp·ωc/10:

```
Kp = J_total·ωc·√(1 + (τcl·ωc)²) / (Kt_eff·√1.01)
Ki = Kp·ωc / 10
```

---

## 6. Method 3 — Root Locus (Pole Placement)

Simplified plant (ignoring τcl lag):

```
G_speed(s) ≈ (Kt_eff/J_total) / s
```

PI + integrator → closed-loop characteristic equation:

```
s² + (Kp·Kt_eff/J_total)·s + Ki·Kt_eff/J_total = 0
```

Matching to 2nd-order form `s² + 2ζωn·s + ωn²` with ωn = 1/τw:

```
Kp = 2·ζ·ωn·J_total / Kt_eff
Ki = ωn²·J_total / Kt_eff
```

Default ζ = 0.707 (minimum overshoot, maximally flat response).

---

## 7. Method 4 — Ziegler-Nichols

Full speed plant with 1.5-sample PWM delay and τcl lag:

```
G_full(s) = (Kt_eff/J_total) / (s·(τcl·s+1)) · (1−0.75Ts·s)/(1+0.75Ts·s)
```

Phase crossover of the P-only loop → ωu, Tu:

```
Kp = 0.45·Ku,   Ki = Kp·1.2/Tu
```

---

## 8. Anti-Windup (Back-Calculation)

When the controller output saturates (e.g. at current limit), the integrator continues to wind up, causing slow recovery. Back-calculation anti-windup feeds the saturation error back to the integrator:

```
integral_dot = error + Kb·(u_saturated − u_unsaturated)
Kb = Ki/Kp   (back-calculation gain = 1/Ti)
```

This limits integrator overshoot when the current limiter activates during large speed steps.

---

## 9. Design Guidelines

| Parameter | Guideline |
|---|---|
| τw (speed) | 5–50 ms typical |
| BWspeed | ≤ BWcurrent / 10 |
| Phase margin | ≥ 45° |
| Gain margin | ≥ 6 dB |
| Anti-windup Kb | Ki/Kp |

The speed loop must never attempt to command iq faster than the current loop can respond. Violating the bandwidth hierarchy causes instability.

---

## References

- Franklin, G.F. et al., *Feedback Control of Dynamic Systems*, Pearson, 2015.
- Bose, B.K., *Modern Power Electronics and AC Drives*, Prentice Hall, 2001.
