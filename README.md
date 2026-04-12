# HuskyAdapt Camera — Hybrid Depth Sensing System

A hybrid proximity sensing system that combines **monocular depth estimation** (MiDaS), **object detection** (YOLOv8), and **ultrasonic sensing** to achieve reliable real-time depth perception on the Clearpath Husky platform.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Host PC (Python)                      │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ YOLOv8 ONNX  │  │ MiDaS Depth  │  │  Ultrasonic  │  │
│  │  Detection   │  │  Estimation  │  │  HTTP Client │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         └────────┬────────┘                  │          │
│           Combined Pipeline ◄────────────────┘          │
└─────────────────────────────────────────────────────────┘
                         │
              BLE / WiFi │
                         │
┌─────────────────────────────────────────────────────────┐
│                ESP32 Peripherals                        │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │ BLE Server 1│  │ BLE Server 2│  │  Ultrasonic    │  │
│  │ Temperature │  │  Humidity   │  │  HTTP Server   │  │
│  └─────────────┘  └─────────────┘  └────────────────┘  │
│        └──────────────┬──────────────┘                  │
│               BLE Client (Hub)                          │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
huskyadaptCamera/
├── vision/                         # Python computer vision
│   ├── detection/
│   │   ├── yolo_detect.py          # YOLOv8 real-time object detection
│   │   └── export_model.py         # Export YOLOv8n .pt → .onnx
│   ├── depth/
│   │   ├── midas_depth.py          # MiDaS monocular depth estimation
│   │   └── midas_grid.py           # Depth with 3×3 spatial grid overlay
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
├── docs/                           # Documentation & media
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

# Depth with 3×3 grid overlay
python vision/depth/midas_grid.py --ref-dist 0.5

# Combined YOLO + MiDaS pipeline (parallel, optimized)
python vision/pipeline/combined_pipeline.py
```

### 4. Flash ESP32 Firmware

Each folder under `firmware/` is a standalone Arduino sketch. Open in Arduino IDE or PlatformIO.

**Required library:** [NimBLE-Arduino](https://github.com/h2zero/NimBLE-Arduino) (for BLE sketches)

| Sketch | Board | Function |
|--------|-------|----------|
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

## Current Progress

- [x] Monocular depth estimation (MiDaS)
- [x] Real-time object detection (YOLOv8n ONNX)
- [x] Combined parallel pipeline with per-object depth
- [x] BLE sensor communication (temperature + humidity)
- [x] Ultrasonic HTTP distance API
- [x] 3×3 spatial depth grid with calibration

## Future Work

- [ ] Integrate ultrasonic sensor into the combined pipeline for absolute reference
- [ ] Sensor fusion (vision + ultrasonic) for improved depth accuracy
- [ ] Point cloud generation from calibrated depth maps
- [ ] ROS integration for Husky platform

## Hardware

- **Compute:** PC/laptop with webcam (CUDA GPU recommended)
- **Microcontrollers:** 3× ESP32 (BLE servers + client)
- **Sensors:** HC-SR04 ultrasonic, DHT22/DS18B20 (temperature), SHT31 (humidity)
- **Platform:** Clearpath Husky UGV

## License

MIT
