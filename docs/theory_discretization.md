# Discretization Theory

## 1. Why Discretization Matters

The PI controller is implemented in a microcontroller running at the PWM interrupt rate (typically 10–20 kHz). The continuous-time gains Kp and Ki must be converted to discrete-time difference equations, and the computational + PWM modulation delay degrades the phase margin.

---

## 2. Continuous-Time PI Controller

```
C(s) = Kp + Ki/s = Kp·(s + Ki/Kp) / s
```

In the time domain:

```
u(t) = Kp·e(t) + Ki·∫e(t)dt
```

---

## 3. Tustin (Bilinear) Discretization

The Tustin method replaces `s` with the bilinear approximation:

```
s = (2/Ts)·(z−1)/(z+1)
```

Pre-warped at the crossover frequency ωc to preserve the gain crossover exactly:

```
s = (ωc/tan(ωc·Ts/2))·(z−1)/(z+1)
```

For the PI integrator `1/s`:

```
1/s → Ts/2 · (z+1)/(z−1)      [trapezoidal rule]
```

Discrete gains:

```
Kp_d = Kp                      [proportional gain unchanged]
Ki_d = Ki · Ts/2               [integrator gain: bilinear approximation]
```

Difference equation (direct form):

```
u[k] = u[k−1] + Kp_d·(e[k] − e[k−1]) + Ki_d·(e[k] + e[k−1])
```

**Advantage:** Preserves frequency-domain shape near ωc. No aliasing of poles. Recommended for closed-loop regulators.

---

## 4. ZOH (Zero-Order Hold) Discretization

Assumes the input is held constant over each sample interval. The exact discrete equivalent of `C(s)` is computed via matrix exponential:

```
C_d(z) = Z{ZOH · C(s)}
```

For the PI integrator:

```
Ki_d ≈ Ki · Ts          [forward Euler approximation of integrator]
```

The ZOH method is exact for the plant but introduces half a sample of additional delay compared to Tustin. Less accurate than Tustin for controller discretization at high Ts·ωc products.

---

## 5. Comparison: Tustin vs ZOH

| Property | Tustin | ZOH |
|---|---|---|
| Integrator mapping | s → 2/Ts·(z−1)/(z+1) | Exact matrix exponential |
| Ki_d | Ki·Ts/2 | Ki·Ts |
| Phase error at ωc | Small (pre-warped) | ~Ts·ωc/2 rad extra |
| Stability | Marginally better | Slightly more conservative |
| Recommendation | Controller design | Plant simulation |

The ratio Ki_d(Tustin)/Ki_d(ZOH) ≈ 0.5 — Tustin integrates more conservatively.

---

## 6. PWM Computation Delay

In a digital current controller the computation completes in the interrupt, but the new duty cycle only takes effect at the next PWM period. This introduces a **1.5-sample delay** (0.5 sample for ZOH of ADC, 1 sample for computation + update):

```
H_delay(s) = e^(−1.5·Ts·s)
```

Approximated by first-order Padé:

```
H_delay(s) ≈ (1 − 0.75·Ts·s) / (1 + 0.75·Ts·s)
```

**Phase loss at crossover:**

```
Δφ = 1.5 · Ts · ωc · (180°/π)     [degrees]
```

At ωc = 2π·500 Hz and Ts = 50 µs:

```
Δφ = 1.5 × 50×10⁻⁶ × 3142 × 57.3 = 13.5°
```

This directly reduces the phase margin. If the continuous design had PM = 90°, the actual PM with delay is ~76.5°.

---

## 7. Maximum Achievable Bandwidth

Given the 45° phase margin requirement and the delay phase loss:

```
BW_max ≈ 1 / (6·Ts)
```

At Ts = 50 µs: BW_max ≈ 3333 Hz.

This is a practical rule of thumb — aggressively tuned loops may approach this limit.

---

## 8. Q15 Fixed-Point Quantization

Microcontrollers without FPU use 16-bit fixed-point arithmetic. Q15 format:

```
Range:      −1 to +1 (minus 1 LSB)
Resolution: 1/32768 ≈ 3×10⁻⁵
Scale:      1 LSB = 1/32768
```

**Scaling procedure:**

1. If Kp > 1: introduce a shift register factor S = 2^⌈log2(Kp)⌉, store Kp/S in Q15.
2. Reconstructed float: Kp_float = Kp_int / (32768 × S).
3. Quantization error: err% = |Kp_float − Kp| / Kp × 100.

Error < 0.01% is achievable for parameters that fit in Q15 without scaling.

**Hex format:** 16-bit two's complement, written as `0xXXXX`.

---

## 9. Firmware-Ready Table

The `firmware_table()` method prints:

```
| Parameter | Continuous | Tustin discrete | Q15 hex  | Q15 float | Error % |
|-----------|------------|-----------------|----------|-----------|---------|
| Kp        | 1.8100     | 1.8100          | 0x7333   | 1.8098    | 0.01%   |
| Ki        | 200.00     | 0.0050          | 0x6666   | 199.99    | 0.003%  |
```

Note: Ki_d (Tustin discrete) = Ki × Ts/2 is the value actually loaded into the firmware difference equation.

---

## 10. Implementation Checklist

- [ ] Use Tustin (pre-warped) for the current loop PI
- [ ] Verify PM after including 1.5·Ts delay
- [ ] Check BW < BW_max = 1/(6·Ts)
- [ ] Quantize Kp and Ki to Q15; verify error < 1%
- [ ] Load Ki_d (not Ki) into firmware difference equation

---

## References

- Franklin, G.F. et al., *Digital Control of Dynamic Systems*, 3rd ed., Addison-Wesley, 1997.
- Holmes, D.G. & Lipo, T.A., *Pulse Width Modulation for Power Converters*, IEEE Press, 2003.
- Buso, S. & Mattavelli, P., *Digital Control in Power Electronics*, Morgan & Claypool, 2006.
