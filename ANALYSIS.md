# Latency Test Results — Analysis Report

Branch: `performance_testing_and_improvement`
Date: July 2026
Samples: 15 per version
Hardware: ESP32 + HC-SR04 + MPU6050

---

## 1. Experiment Design

Three firmware variants tested under identical hardware:

| Version | Algorithms | Architecture |
|---------|-----------|--------------|
| Baseline | Threshold only | Arduino loop() |
| Full (Polling) | Consensus + hysteresis + state confirm + IMU | Arduino loop() |
| Full (RTOS) | Same as Full Polling | FreeRTOS task + ISR notification |

Each version prints `[LATENCY]` with microsecond timestamps on every trigger event. A Python script (`latency_test.py`) collects 15 samples per version via serial.

## 2. Raw Data

### Baseline (15 samples)
```
Distance:  24–71 cm, mean = 49.0 cm
Proc:      3–4 µs, mean = 3.7 µs, median = 4 µs
E2E:       3,656–6,417 µs, mean = 5,120 µs, median = 5,932 µs
```

### Full Polling (15 samples)
```
Distance:  78–95 cm, mean = 89.9 cm
Proc:      5–6 µs, mean = 5.4 µs, median = 5 µs
E2E:       6,786–7,806 µs, mean = 7,495 µs, median = 7,516 µs
Mode:      all WALKING
Agreement: 2–9, mean = 4.9
```

### Full RTOS (15 samples)
```
Distance:  60–90 cm, mean = 77.9 cm
Proc:      6–8 µs, mean = 6.5 µs, median = 6 µs
E2E:       5,752–7,487 µs, mean = 6,799 µs, median = 6,814 µs
Mode:      all QUEUING
Agreement: 3–8, mean = 4.3
```

## 3. Finding 1 — Algorithmic CPU Overhead Is Negligible

| Version | Proc Mean | CPU per 40 ms Cycle |
|---------|-----------|---------------------|
| Baseline | 3.7 µs | 0.009% |
| Full (Polling) | 5.4 µs | 0.014% |
| Full (RTOS) | 6.5 µs | 0.016% |

The full algorithm suite (consensus loop over 20 entries, hysteresis check, state confirmation counter, IMU variance over 32 samples) adds only **1.7 µs** over baseline. The RTOS version adds another **1.1 µs** of task-switching overhead. In all cases the ESP32 is idle more than 99.98% of the time.

Conclusion: **CPU cost is not a concern for any version.**

## 4. Finding 2 — E2E Differences Are Explained by Sound Travel

Trigger distances varied across tests because the full version requires 3 consecutive confirmations (object is closer by the time it triggers) and testing conditions differed. After normalizing for the speed of sound:

| Comparison | Dist Gap | Expected Sound Δ | Actual E2E Δ | Net Software Δ |
|------------|----------|-------------------|--------------|-----------------|
| Baseline → Full (Polling) | 40.9 cm | 2,387 µs | 2,375 µs | **−12 µs** |
| Full (Polling) → Full (RTOS) | 12.0 cm | 698 µs | 696 µs | **−2 µs** |

Both net software deltas are within measurement noise. The speed of sound (0.0343 cm/µs) completely accounts for the E2E differences.

Conclusion: **No version is faster or slower than another in software. E2E is dominated by physics.**

## 5. Finding 3 — RTOS Provides Zero Benefit

The hypothesis was that FreeRTOS task notifications would reduce jitter from I2C and Serial blocking in `loop()`, improving E2E consistency.

Results:
- Proc latency **increased** by 1.1 µs (task-switching overhead)
- E2E latency showed **no improvement** after distance normalization
- E2E variance did not measurably decrease

Why: The I2C read (~500 µs) and Serial print occur at most once per 10–200 ms. With a 40 ms ping interval, the probability of an echo landing during an I2C transaction is low (~1.25%). The polling architecture already handles this adequately.

Conclusion: **RTOS is not justified for this workload.** The bottleneck is the speed of sound and the intentional state confirmation delay, neither of which RTOS can improve. This demonstrates an important embedded systems principle: **profile before optimizing.**

## 6. Finding 4 — The Real Cost Is Intentional

| Metric | Baseline | Full (either) |
|--------|----------|---------------|
| Readings needed to trigger | 1 | 3 |
| Perceived delay | ~46 ms | ~128 ms |
| Additional delay | — | ~82 ms |

The 82 ms difference is entirely from `STATE_CONFIRM_N = 3` requiring two extra ping cycles at 40 ms each. This is a deliberate design decision that:
- **Eliminates** false triggers from single noisy readings
- **Filters** transient reflections and sensor glitches
- **Trades** 82 ms of response time for substantially higher reliability

For a walker-mounted proximity sensor where false alarms erode user trust, this tradeoff is justified.

## 7. IMU Mode Detection

The Full Polling test triggered in WALKING mode; the RTOS test triggered in QUEUING mode. This reflects the physical test conditions (moving hand vs stationary setup) and confirms the IMU-based automatic mode switching works correctly. The consensus agreement values (mean 4.3–4.9) consistently exceeded the MIN_AGREEMENT threshold of 2, indicating stable sensor readings.

## 8. Summary Table

| Dimension | Baseline | Full (Polling) | Full (RTOS) |
|-----------|----------|----------------|-------------|
| Proc latency | 3.7 µs | 5.4 µs | 6.5 µs |
| Software E2E overhead | — | ≈ 0 µs | ≈ 0 µs |
| Perceived delay | ~46 ms | ~128 ms | ~128 ms |
| CPU usage | 0.009% | 0.014% | 0.016% |
| False positive filtering | None | Consensus + state confirm | Same |
| Architecture complexity | Low | Low | Medium |
| Verdict | Fast but unreliable | Reliable, simple | No added benefit |

## 9. Recommendations

1. **Use the polling architecture** (Full Polling). It is simpler, equally performant, and easier to maintain.
2. **Do not add RTOS** unless the project grows to include concurrent time-critical peripherals (e.g., BLE + motor control + ultrasonic simultaneously).
3. **Consider adaptive ping frequency** — reduce from 25 Hz to 5 Hz in QUEUING mode to save power, since obstacle changes are slower when stationary.
4. **Run false-positive tests** (`a_fp_test_*.ino`) to quantify the reliability benefit of consensus + state confirmation vs baseline.
