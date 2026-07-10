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

## Expected Results

| Metric | Baseline | Full | Why |
|--------|----------|------|-----|
| Processing latency | ~2–3 µs | ~5–20 µs | Full version runs consensus loop over 20-element history + IMU variance computation |
| E2E (single cycle) | ~15 ms | ~15 ms | Dominated by speed of sound, not software |
| True perceived delay | ~15 ms | ~95 ms | Full version requires 3 confirmations × 40ms ping interval |
| False positives | Higher | Lower | Consensus + state confirmation filters noise |

The key tradeoff: the algorithms add negligible CPU cost but intentionally delay the decision by ~80ms to filter out false triggers.

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
