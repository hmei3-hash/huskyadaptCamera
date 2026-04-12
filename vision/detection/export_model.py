"""
Export YOLOv8n weights to ONNX format.

Usage:
    python export_model.py [--size 320] [--output ../models/yolov8n.onnx]

This downloads yolov8n.pt (if needed) and exports it as an ONNX model.
"""

import os
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8n to ONNX")
    parser.add_argument("--size", type=int, default=320, help="Input image size")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output or os.path.join(script_dir, "..", "models")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading YOLOv8n weights...")
    model = YOLO("yolov8n.pt")

    print(f"Exporting to ONNX (imgsz={args.size})...")
    onnx_path = model.export(format="onnx", imgsz=args.size, dynamic=False)

    print(f"Export complete: {onnx_path}")


if __name__ == "__main__":
    main()
