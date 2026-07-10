# Performance Testing & Improvement

Branch: `performance_testing_and_improvement`

Latency and performance benchmarking for the HuskyAdapt proximity sensor. Compares a stripped-down baseline (threshold only) against the full IMU fusion algorithm (consensus + hysteresis + state confirmation + IMU auto mode switching) to quantify the cost of each algorithm layer.

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
├── a_baseline_latency.ino      # Baseline firmware — threshold only, no algorithms
├── a_full_imu_latency.ino      # Full firmware — consensus, hysteresis, state confirm, IMU
├── latency_test.py             # Python serial data collector + stats + comparison
├── latency_baseline.csv        # Collected baseline data (15 samples)
├── latency_full.csv            # Collected full version data (15 samples)
├── latency_comparison.txt      # Markdown comparison table
├── ANALYSIS.md                 # Detailed results analysis and findings
└── README.md
```

## What Each Firmware Does

**Baseline (`a_baseline_latency.ino`)**

Simplest possible detection: if `distance < 100cm` and was previously outside, fire trigger immediately. No consensus, no history buffer, no hysteresis, no state confirmation, no IMU. One reading is enough to trigger.

**Full (`a_full_imu_latency.ino`)**

All algorithm layers enabled:
- 20-reading history buffer with consensus filter (requires 2+ similar readings)
- 10cm hysteresis band to prevent edge flickering
- State confirmation requiring 3 consecutive readings before committing
- MPU6050 accelerometer variance for automatic WALKING/QUEUING mode switching
- Mode-dependent re-arm bands (110cm walking, 150cm queuing)

Both firmwares print `[LATENCY]` lines on each trigger event with microsecond timestamps.

## Prerequisites

```bash
pip install pyserial
```

Arduino IDE with ESP32 board support installed.

## Step-by-Step Test Procedure

### Step 1: Find your serial port

Plug in the ESP32 via USB and run:

```bash
python -m serial.tools.list_ports
```

Note the port (e.g. `COM8` on Windows, `/dev/ttyUSB0` on Linux, `/dev/cu.usbserial-xxxx` on Mac).

### Step 2: Collect baseline data

1. Open `a_baseline_latency.ino` in Arduino IDE.
2. Select your board and port under **Tools → Board** and **Tools → Port**.
3. Click **Upload**.
4. **Close the Serial Monitor** (Python and Serial Monitor cannot share the port).
5. Run:

```bash
python latency_test.py --port COM8 --label baseline --samples 15
```

6. Place your hand far from the sensor (>100cm). Move it close (<100cm) to trigger. Move it away again until the serial output shows `inside=N`. Repeat 15 times. The script collects data automatically and saves to `latency_baseline.csv`.

### Step 3: Collect full algorithm data

1. Open `a_full_imu_latency.ino` in Arduino IDE.
2. Upload.
3. Close Serial Monitor.
4. Run:

```bash
python latency_test.py --port COM8 --label full --samples 15
```

5. Same procedure: hand in, wait for trigger, hand out, wait for reset, repeat 15 times. Saves to `latency_full.csv`.

### Step 4: Generate comparison

```bash
python latency_test.py --compare
```

Outputs a comparison table to the terminal and saves a markdown table to `latency_comparison.txt`.

## Metrics Explained

| Metric | What It Measures |
|--------|-----------------|
| `proc` (processing latency) | Time from echo received to GPIO pulled HIGH. Pure software execution time of `processReading()`. |
| `e2e` (end-to-end latency) | Time from ultrasonic ping sent to GPIO pulled HIGH. Includes sound travel + processing. |
| True perceived delay | Time from object entering range to alert firing. For baseline this equals `e2e`. For full version this equals `e2e + (STATE_CONFIRM_N - 1) × PING_INTERVAL_MS` because the algorithm waits for 3 consecutive confirming readings. |

## Results

Tested July 2026, 15 samples per version. Full analysis in [`ANALYSIS.md`](ANALYSIS.md).

| Metric | Baseline | Full | Delta |
|--------|----------|------|-------|
| Proc latency (median) | 4 µs | 5 µs | +1 µs |
| Proc latency (mean) | 3.7 µs | 5.4 µs | +1.7 µs |
| E2E latency (median) | 5,932 µs | 7,516 µs | +1,584 µs |
| E2E latency (mean) | 5,120 µs | 7,495 µs | +2,375 µs |
| True perceived delay | ~46 ms | ~128 ms | +82 ms |
| CPU usage per cycle | 0.009% | 0.014% | negligible |

**Key findings:**

1. **Algorithmic CPU cost is negligible.** Consensus + hysteresis + state confirmation + IMU variance together add only 1.7 µs.
2. **E2E difference is physics, not software.** The full version triggers near the 100 cm threshold (~90 cm) while the baseline triggers at closer range (~49 cm). After normalizing for sound travel distance, software overhead is effectively zero.
3. **The real cost is intentional delay.** `STATE_CONFIRM_N = 3` adds ~80 ms of deliberate waiting to filter false positives. This is a design tradeoff, not computational overhead.

## Tips

- Keep the test distance consistent across runs (e.g. always trigger at ~50cm) for fair comparison.
- Wait for `inside=N` in the serial output before triggering the next sample.
- If `dist=0` appears frequently, check your ultrasonic wiring.
- If `var=0.0000` in the full version, check MPU6050 I2C wiring (SDA→40, SCL→41).
- Replace `COM8` with your actual port in all commands.

## Reproducing from Scratch

```bash
# Clone and switch branch
git clone https://github.com/hmei3-hash/huskyadaptProximitySensor.git
cd huskyadaptProximitySensor
git checkout performance_testing_and_improvement

# Install Python dependency
pip install pyserial

# Flash baseline via Arduino IDE, then:
python latency_test.py --port COM8 --label baseline --samples 15

# Flash full version via Arduino IDE, then:
python latency_test.py --port COM8 --label full --samples 15

# Compare
python latency_test.py --compare
```
