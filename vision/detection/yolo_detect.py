"""
YOLOv8 Real-Time Object Detection
==================================
Runs YOLOv8n (ONNX) on a webcam feed with bounding boxes and class labels.

Prerequisites:
    Run `python export_model.py` first to generate yolov8n.onnx

Usage:
    python yolo_detect.py [--size 320] [--conf 0.5]

Keys:
    Q - quit
"""

import os
import argparse
import cv2 as cv
import numpy as np
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 webcam detection")
    parser.add_argument("--size", type=int, default=320, help="Input size (default: 320)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--model", default=None, help="Path to ONNX model")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model or os.path.join(SCRIPT_DIR, "..", "models", "yolov8n.onnx")

    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Run `python export_model.py` first.")
        return

    net = cv.dnn.readNetFromONNX(model_path)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    fps_start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_count += 1

        # FPS reporting every 2 seconds
        elapsed = time.time() - fps_start
        if elapsed >= 2.0:
            print(f"FPS: {frame_count / elapsed:.2f}")
            frame_count = 0
            fps_start = time.time()

        # Inference
        blob = cv.dnn.blobFromImage(
            frame, 1 / 255.0, (args.size, args.size), swapRB=True, crop=False
        )
        net.setInput(blob)
        output = net.forward()[0]

        if output.shape[0] < output.shape[1]:
            output = output.T  # -> (N, 84)

        # Vectorized parsing
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(scores)), class_ids]
        mask = confs >= args.conf

        if np.any(mask):
            det = output[mask]
            det_confs = confs[mask].astype(np.float32)
            det_cls = class_ids[mask]

            cx, cy, bw, bh = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
            sx, sy = w / args.size, h / args.size
            x1 = ((cx - bw / 2) * sx).astype(int).clip(0, w - 1)
            y1 = ((cy - bh / 2) * sy).astype(int).clip(0, h - 1)
            x2 = ((cx + bw / 2) * sx).astype(int).clip(0, w - 1)
            y2 = ((cy + bh / 2) * sy).astype(int).clip(0, h - 1)

            bw_px = (x2 - x1).clip(0)
            bh_px = (y2 - y1).clip(0)
            valid = (bw_px > 0) & (bh_px > 0)

            boxes = np.stack([x1[valid], y1[valid], bw_px[valid], bh_px[valid]], axis=1).tolist()
            confs_list = det_confs[valid].tolist()
            cls_list = det_cls[valid].tolist()

            indices = cv.dnn.NMSBoxes(boxes, confs_list, args.conf, args.nms)
            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bbw, bbh = boxes[idx]
                    cls_id = int(cls_list[idx])
                    conf = confs_list[idx]
                    name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"id_{cls_id}"

                    cv.rectangle(frame, (bx, by), (bx + bbw, by + bbh), (0, 255, 0), 2)
                    label = f"{name} ({conf:.2f})"
                    cv.putText(frame, label, (bx, by - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv.imshow("YOLO Detection", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
