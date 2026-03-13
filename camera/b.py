import os
from ultralytics import YOLO

print("CWD:", os.getcwd())

try:
    print("Loading YOLOv8n .pt weights...")
    model = YOLO("yolov8n.pt")  # will download if not present
    print("Model loaded.")

    print("Exporting to ONNX (this can take a bit)...")
    onnx_path = model.export(format="onnx", imgsz=320, dynamic=False)

    print("Export finished. ONNX saved to:", onnx_path)

except Exception as e:
    print("ERROR during export:", repr(e))
