# HuskyAdapt Proximity Sensor

A proximity sensing system for the Clearpath Husky walker platform that detects nearby obstacles and alerts the user via vibration/buzzer. The project explores multiple sensing approaches — from ultrasonic-only firmware to hybrid camera-based depth estimation — across different branches.

## Approaches

This project implements **5 distinct approaches** across branches, each tackling proximity detection differently:

| # | Branch | Approach | Key Idea |
|---|--------|----------|----------|
| 1 | `main` | **Consensus + Hysteresis + Dual Mode** | Maintains a 20-reading history buffer; requires consensus (multiple similar readings) before triggering. Uses hysteresis band to prevent flickering at the threshold edge. Supports QUEUE (trigger once per entry) and STRICT (continuous trigger while inside) modes, switchable via button. State confirmation requires N consecutive readings before changing inside/outside status. |
| 2 | `less_logic_branch` | **Consensus + Stillness Detection** | Same consensus filter as `main`, but replaces mode switching with automatic stillness detection. A separate 30-reading window checks whether the object has been stationary (readings within 8 cm range). If still, the trigger is suppressed — solving the problem of alerting on walls or stationary furniture. |
| 3 | `median-logic-branch` | **Median Filter + Multi-Mode** | Replaces the consensus algorithm with a 5-sample median filter for noise rejection. Adds a third LINE mode (in addition to QUEUE and STRICT) that requires more consecutive clear readings before re-arming — better for slow, uneven movement like queuing in a line. Cleanly separates sensor reading, presence detection, and trigger output into distinct functions. |
| 4 | `imu_fusion` | **IMU-Based Automatic Mode Switching** | Fuses ultrasonic distance with MPU6050 accelerometer data. Computes variance of accelerometer magnitude over a 32-sample window to automatically distinguish WALKING vs QUEUING modes — no button needed. Walking mode uses a tight re-arm band (110 cm), while queuing mode uses a wide release band (150 cm) so the sensor stays quiet during normal queue shuffling. Mode switching is debounced with a 2-second hold timer. |
| 5 | `main` (vision/) | **Camera: MiDaS + YOLOv8 + Ultrasonic** | Host-PC Python pipeline combining monocular depth estimation (MiDaS), object detection (YOLOv8 ONNX), and ultrasonic HTTP readings. YOLO and MiDaS run in parallel via ThreadPoolExecutor. Provides per-object depth estimation and a 3x3 spatial depth grid overlay. |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Host PC (Python Vision)                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ YOLOv8 ONNX  │  │ MiDaS Depth  │  │   Ultrasonic      │  │
│  │  Detection   │  │  Estimation  │  │   HTTP Client     │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘  │
│         └────────┬────────┘                    │             │
│           Combined Pipeline ◄──────────────────┘             │
└──────────────────────────────────────────────────────────────┘
                         │
              BLE / WiFi │
                         │
┌──────────────────────────────────────────────────────────────┐
│                   ESP32 Peripherals                          │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │
│  │ BLE Temp │  │ BLE Hum  │  │ Ultrasonic│  │ IMU       │  │
│  │ Server   │  │ Server   │  │ HTTP Srv  │  │ (MPU6050) │  │
│  └────┬─────┘  └────┬─────┘  └───────────┘  └───────────┘  │
│       └──────┬───────┘                                       │
│         BLE Client (Hub)                                     │
└──────────────────────────────────────────────────────────────┘
                         │
                    Trigger Pin
                         │
              ┌──────────▼──────────┐
              │  Vibration / Buzzer │
              └─────────────────────┘
```

## Project Structure

```
huskyadaptProximitySensor/
├── a.ino                           # Main ultrasonic firmware (see branches for variants)
├── find_dist.py                    # Standalone YOLOv8-pose + MiDaS depth script
├── vision/                         # Python computer vision pipeline
│   ├── detection/
│   │   ├── yolo_detect.py          # YOLOv8 real-time object detection
│   │   └── export_model.py         # Export YOLOv8n .pt → .onnx
│   ├── depth/
│   │   ├── midas_depth.py          # MiDaS monocular depth estimation
│   │   └── midas_grid.py           # Depth with 3x3 spatial grid overlay
│   └── pipeline/
│       └── combined_pipeline.py    # YOLO + MiDaS parallel pipeline
│
├── firmware/                       # ESP32 Arduino sketches
│   ├── ble_client/                 # BLE hub — connects to both servers
│   ├── ble_server_temp/            # BLE peripheral — temperature sensor
│   ├── ble_server_humidity/        # BLE peripheral — humidity sensor
│   └── ultrasonic_http/            # WiFi HTTP server + HC-SR04 sensor
│
├── models/                         # Model weights (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start

### 1. Python Environment

```bash
pip install -r requirements.txt
```

### 2. Export YOLOv8 Model

```bash
cd vision/detection
python export_model.py          # downloads yolov8n.pt → exports yolov8n.onnx
```

The ONNX file is saved to `models/` and is git-ignored.

### 3. Run Individual Modules

```bash
# Object detection only
python vision/detection/yolo_detect.py

# Depth estimation only
python vision/depth/midas_depth.py --ref-dist 0.5

# Depth with 3x3 grid overlay
python vision/depth/midas_grid.py --ref-dist 0.5

# Combined YOLO + MiDaS pipeline (parallel, optimized)
python vision/pipeline/combined_pipeline.py
```

### 4. Flash ESP32 Firmware

Each folder under `firmware/` is a standalone Arduino sketch. Open in Arduino IDE or PlatformIO. The root `a.ino` is the main ultrasonic proximity firmware — switch branches to try different algorithms.

**Required libraries:**
- [NimBLE-Arduino](https://github.com/h2zero/NimBLE-Arduino) (for BLE sketches)
- Wire.h (for IMU fusion branch, built-in)

| Sketch | Board | Function |
|--------|-------|----------|
| `a.ino` | ESP32 | Main proximity detector — consensus, hysteresis, dual mode |
| `ble_server_temp/` | ESP32 | Advertises as "TempSensor_1", sends temperature via BLE NOTIFY |
| `ble_server_humidity/` | ESP32 | Advertises as "HumSensor_2", sends humidity via BLE NOTIFY |
| `ble_client/` | ESP32 | Connects to both BLE servers, aggregates sensor data |
| `ultrasonic_http/` | ESP32 | HC-SR04 ultrasonic sensor + WiFi HTTP API at `/reading` |

## Combined Pipeline Features

The `combined_pipeline.py` is the main entry point with these optimizations:

- **Parallel inference** — YOLO and MiDaS run concurrently via `ThreadPoolExecutor`
- **Frame skipping** — configurable N-frame interval (press `+`/`-` at runtime)
- **Vectorized YOLO parsing** — no Python for-loop over detections
- **Reduced resolution** — YOLO@320, MiDaS@192 for speed
- **FP16 on CUDA** — automatic half-precision when GPU is available
- **Strided grid sampling** — skips pixels when computing cell averages
- **Per-object depth** — each detected object shows estimated distance

### Runtime Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `+` / `-` | Increase / decrease frame skip |
| Left-click | Print depth at clicked pixel |

## Hardware

- **Compute:** PC/laptop with webcam (CUDA GPU recommended for vision pipeline)
- **Microcontrollers:** 3x ESP32 (BLE servers + client)
- **Sensors:** HC-SR04 ultrasonic, DHT22/DS18B20 (temperature), SHT31 (humidity), MPU6050 (IMU, for `imu_fusion` branch)
- **Platform:** Clearpath Husky walker
- **Output:** Vibration motor or buzzer connected to trigger pin

## License

MIT
