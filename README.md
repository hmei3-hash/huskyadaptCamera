# Performance Testing & Improvement

Branch: `performance_testing_and_improvement`

Latency, false-positive, and architecture benchmarking for the HuskyAdapt proximity sensor. Compares three firmware variants to quantify the cost and benefit of each algorithm layer, including an RTOS experiment that proved the bottleneck is physics, not software.

## Hardware Setup

| Component | GPIO | Notes |
|-----------|------|-------|
| HC-SR04 TRIG | 5 | Ultrasonic trigger |
| HC-SR04 ECHO | 4 | Ultrasonic echo |
| LED / Buzzer | 6 | Alert output (TRIGGER_PIN) |
| MPU6050 SDA | 40 | I2C data (full version only) |
| MPU6050 SCL | 41 | I2C clock (full version only) |

Board: ESP32

## Files

```
performance_testing_and_improvement/
│
├── Latency Testing
│   ├── a_baseline_latency.ino          # Baseline — threshold only, no algorithms
│   ├── a_full_imu_latency.ino          # Full — consensus + hysteresis + state confirm + IMU
│   ├── a_step1_task_notification.ino   # Full + FreeRTOS task notification (RTOS experiment)
│   ├── latency_test.py                 # Python serial collector + stats + comparison
│   ├── latency_baseline.csv            # Baseline data (15 samples)
│   ├── latency_full.csv                # Full polling data (15 samples)
│   ├── latency_rtos.csv                # Full RTOS data (15 samples)
│   └── latency_comparison.txt          # Baseline vs Full comparison table
│
├── False Positive Testing
│   ├── a_fp_test_baseline.ino          # Baseline — 3-min auto-counting test
│   └── a_fp_test_full.ino              # Full — 3-min auto-counting test with consensus stats
│
├── ANALYSIS.md                         # Detailed results and findings
└── README.md
```

## Three Firmware Variants

### Baseline (`a_baseline_latency.ino`)
Simplest possible detection. If `distance < 100cm` and was previously outside, fire trigger immediately. No consensus, no history, no hysteresis, no state confirmation, no IMU. One reading triggers.

### Full Polling (`a_full_imu_latency.ino`)
All algorithm layers enabled in a traditional Arduino `loop()`:
- 20-reading history buffer with consensus filter (MIN_AGREEMENT = 2)
- 10 cm hysteresis band to prevent edge flickering
- State confirmation requiring 3 consecutive readings
- MPU6050 accelerometer variance for auto WALKING/QUEUING mode switching
- Mode-dependent re-arm bands (110 cm walking, 150 cm queuing)

### Full RTOS (`a_step1_task_notification.ino`)
Same algorithms as Full Polling, but restructured with FreeRTOS:
- `echoISR()` sends a Task Notification instead of setting a flag
- `ultrasonicTask` (priority 3) blocks until notified, preempts `loop()`
- `loop()` (priority 1) only handles IMU, pulse drop, and serial print
- Hypothesis: reducing polling jitter would improve E2E latency

## Results

### Latency Comparison (15 samples each)

| Metric | Baseline | Full (Polling) | Full (RTOS) |
|--------|----------|----------------|-------------|
| Proc latency (median) | 4 µs | 5 µs | 6 µs |
| Proc latency (mean) | 3.7 µs | 5.4 µs | 6.5 µs |
| E2E latency (median) | 5,932 µs | 7,516 µs | 6,814 µs |
| E2E latency (mean) | 5,120 µs | 7,495 µs | 6,799 µs |
| Avg trigger distance | 49.0 cm | 89.9 cm | 77.9 cm |
| Mode at trigger | — | WALKING | QUEUING |
| CPU usage per cycle | 0.009% | 0.014% | 0.016% |

### Distance-Normalized Analysis

E2E latency is dominated by the speed of sound. Trigger distances varied across tests, so raw E2E numbers are not directly comparable. After normalizing:

| Comparison | Distance Gap | Expected Sound Δ | Actual E2E Δ | Software Overhead |
|------------|-------------|-------------------|--------------|-------------------|
| Baseline vs Full (Polling) | 40.9 cm | 2,387 µs | 2,375 µs | ≈ 0 µs |
| Full (Polling) vs Full (RTOS) | 12.0 cm | 698 µs | 696 µs | ≈ 0 µs |

In both cases, after accounting for sound travel distance, the software overhead is effectively **zero**.

### True User-Perceived Delay

| Version | Formula | Typical Delay |
|---------|---------|---------------|
| Baseline | 1 ping cycle + E2E | ~46 ms |
| Full | 3 ping cycles + E2E | ~128 ms |
| Full (RTOS) | 3 ping cycles + E2E | ~128 ms |

The 82 ms difference between baseline and full comes entirely from `STATE_CONFIRM_N = 3` requiring two extra 40 ms ping cycles. RTOS does not reduce this.

## Key Findings

### 1. Algorithmic CPU cost is negligible
Consensus (20-element loop) + hysteresis + state confirmation + IMU variance together add only 1.7 µs. The ESP32 is idle >99.98% of the time.

### 2. E2E differences are physics, not software
After normalizing for trigger distance, the software overhead between all three versions is within measurement noise (≈ 0 µs). The speed of sound dominates.

### 3. RTOS does not help this project
The RTOS experiment added 1 µs of task-switching overhead to proc latency and provided zero improvement to E2E. The bottleneck is the speed of sound and the intentional state confirmation delay, neither of which RTOS can address. This is a valid finding: **correct performance analysis means knowing when not to optimize.**

### 4. The real cost is a design choice
The full version's ~82 ms extra perceived delay is intentional — it waits for 3 consecutive confirming readings to filter false triggers. This trades response speed for reliability.

## False Positive Testing

Two firmwares for automated false-positive measurement over 3-minute timed runs.

### Test Procedure

Each scenario runs for 3 minutes with nothing moving. The firmware auto-counts triggers and prints a summary.

| Scenario | Setup | Expected Triggers |
|----------|-------|-------------------|
| A. Open air | No obstacle within 200 cm | 0 |
| B. Wall at 80 cm | Sensor facing wall, 80 cm away | 1 (initial only) |
| C. Furniture at 60 cm | Chair or table at 60 cm | 1 (initial only) |
| D. Threshold edge ~100 cm | Obstacle right at threshold boundary | 0 or 1 |

```bash
# Flash a_fp_test_baseline.ino or a_fp_test_full.ino
# Open Serial Monitor at 115200
# Wait 3 minutes — do not touch anything
# Copy the printed summary block
```

The full version additionally reports `rejectedByConsensus` — how many inside-threshold readings the consensus filter blocked.

## Reproducing the Latency Tests

### Prerequisites

```bash
pip install pyserial
```

### Step-by-step

```bash
# Clone and switch branch
git clone https://github.com/hmei3-hash/huskyadaptProximitySensor.git
cd huskyadaptProximitySensor
git checkout performance_testing_and_improvement

# Install Python dependency
pip install pyserial

# Find your serial port
python -m serial.tools.list_ports

# --- Test 1: Baseline ---
# Flash a_baseline_latency.ino via Arduino IDE
# Close Serial Monitor, then:
python latency_test.py --port COM8 --label baseline --samples 15
# Move hand in (<100cm) and out (>100cm) 15 times

# --- Test 2: Full (polling) ---
# Flash a_full_imu_latency.ino via Arduino IDE
# Close Serial Monitor, then:
python latency_test.py --port COM8 --label full --samples 15

# --- Test 3: Full (RTOS) ---
# Flash a_step1_task_notification.ino via Arduino IDE
# Close Serial Monitor, then:
python latency_test.py --port COM8 --label rtos --samples 15

# --- Compare baseline vs full ---
python latency_test.py --compare
```

Replace `COM8` with your actual serial port.

### Tips
- Keep trigger distance consistent across tests for fair comparison
- Wait for `inside=N` before triggering the next sample
- If `dist=0` appears frequently, check ultrasonic wiring
- If `var=0.0000` in full version, check MPU6050 I2C wiring (SDA→40, SCL→41)
- Close Arduino Serial Monitor before running the Python script
