# Current Loop Tuning Theory

## 1. Control Structure

Each d-q axis current is regulated by an independent PI controller. Cross-coupling and back-EMF are cancelled by feedforward terms so each axis sees a first-order plant:

```
G(s) = (1/Rs) / (τ·s + 1)      τ = L/Rs
```

PI controller (parallel form):

```
C(s) = Kp + Ki/s = Kp·(s + Ki/Kp) / s
```

Open-loop transfer function:

```
L(s) = C(s)·G(s) = (Kp·s + Ki) / (Rs·s·(τ·s + 1))
```

Closed-loop:

```
T(s) = L(s)/(1+L(s)) = (Kp·s + Ki) / (Rs·τ·s² + (Rs+Kp)·s + Ki)
```

Unity DC gain is guaranteed by the integrator in C(s).

---

## 2. Method 1 — Pole-Zero Cancellation

**Principle:** Place the PI zero to cancel the plant pole, then set the gain for the desired closed-loop bandwidth.

```
Plant pole:   s = −Rs/L = −1/τ
PI zero:      s = −Ki/Kp  →  set Ki/Kp = Rs/L

Desired CL pole: s = −1/τcl

=> Kp = L / τcl
   Ki = Rs / τcl
```

After cancellation the closed-loop simplifies to a first-order system:

```
T(s) ≈ 1 / (τcl·s + 1)
```

**Tuning knob:** τcl. Typical value: 5–10 × Ts_PWM (e.g. 0.5–1 ms at 20 kHz).

**Phase margin:** ~90° (exact cancellation gives near-infinite GM, PM ≈ 90°).

---

## 3. Method 2 — Frequency Domain

**Principle:** Specify the desired gain crossover frequency ωc and solve for Kp.

At crossover `|C(jωc)·G(jωc)| = 1`.

With Ki = Kp·ωc/10 (integral 1 decade below crossover):

```
|C(jωc)| ≈ Kp·√(1 + 1/100) ≈ Kp·√1.01

|G(jωc)| = (1/Rs) / √(1 + (τ·ωc)²)

=> Kp = Rs·√(1 + (τ·ωc)²) / √1.01
   Ki = Kp·ωc / 10
```

**Tuning knob:** BW_Hz. Achievable up to ≈ 1/(6·Ts) before PWM delay erodes phase margin.

---

## 4. Method 3 — Root Locus (Pole Placement)

**Principle:** Match the closed-loop characteristic equation to a desired second-order form.

Closed-loop denominator (from L(s) above):

```
Rs·τ·s² + (Rs + Kp)·s + Ki = 0
```

Dividing by Rs·τ and matching to `s² + 2ζωn·s + ωn² = 0`:

```
ωn = 1/τcl      (set desired natural frequency)

Kp = Rs·(2·ζ·ωn·τ − 1)
Ki = ωn²·Rs·τ
```

If the expression for Kp ≤ 0 (ωn too small), ωn is raised to the minimum value that yields Kp > 0:

```
ωn_min = 1 / (2·ζ·τ)
```

**Tuning knobs:** τcl and target_zeta (default 0.707 for critically damped).

---

## 5. Method 4 — Ziegler-Nichols

**Principle:** Simulate P-only closed loop with 1.5-sample PWM delay. Find ultimate gain Ku and period Tu, then apply Z-N PI formulas.

**1.5-sample Padé delay:**

```
e^(−1.5·Ts·s) ≈ (1 − 0.75·Ts·s) / (1 + 0.75·Ts·s)
```

Combined plant+delay phase crosses −180° at ultimate frequency ωu:

```
Tu = 2π / ωu
Ku = 1 / |G_delay(jωu)|

Kp = 0.45·Ku       (Z-N PI formula)
Ki = Kp·1.2 / Tu
```

Z-N gives aggressive tuning — typically lower phase margin than other methods. Use as a comparison benchmark.

---

## 6. Verification Suite

All four methods produce the same `LoopResult` populated by the same verification code:

| Metric | How computed |
|---|---|
| crossover_Hz | Frequency where \|L(jω)\| = 0 dB |
| PM_deg | 180° + ∠L(jωc) |
| GM_dB | −\|L(jωp)\| at phase crossover ωp |
| BW_Hz | Closed-loop −3 dB frequency |
| settling_ms | Last time y(t) exits 2% band |
| overshoot_pct | (peak/y_final − 1) × 100 |

Warnings raised when PM < 45°, GM < 6 dB.

---

## 7. Design Guidelines

| Parameter | Guideline |
|---|---|
| τcl (pole-zero) | 5–10 × Ts_PWM |
| BW_Hz (freq. domain) | ≤ 1/(6·Ts) ≈ 3.3 kHz at 20 kHz |
| Phase margin | ≥ 45° |
| Gain margin | ≥ 6 dB |
| Settling time | ≤ 5·τcl |

For SPMSM: d and q axes have identical gains (Ld = Lq).  
For IPMSM: q-axis gains differ because Lq > Ld.

---

## References

- Åström, K.J. & Hägglund, T., *PID Controllers: Theory, Design and Tuning*, ISA, 1995.
- Mohan, N., *Advanced Electric Drives*, Wiley, 2014.
