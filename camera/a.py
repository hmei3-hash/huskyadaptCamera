import cv2 as cv
import numpy as np
import time

start_time = time.time()
# COCO class names (index 0 = "person", 1 = "bicycle", ...)
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
    "hair drier", "toothbrush"
]

# 1. Load ONNX model
net = cv.dnn.readNetFromONNX("yolov8n.onnx")

# 2. Open camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count+=1
    h, w = frame.shape[:2]
    if time.time() - start_time >= 2.0:
        fps = frame_count / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    # 3. Create blob
    blob = cv.dnn.blobFromImage(
    frame,
    scalefactor=1 / 255.0,
    size=(320, 320),  # 改成 416x416
    swapRB=True,
    crop=False
)
    net.setInput(blob)

    # 4. Forward pass
    output = net.forward()  # shape ~ (1, 84, 8400) or (1, 8400, 84)

    out = output[0]  # get the first result. we are only analyzing one image
    if out.shape[0] < out.shape[1]:
        out = out.T  # ensure (num_dets, 84)

    num_channels = out.shape[1]
    if num_channels != 84:
        print(f"Unexpected channels: {num_channels}")
        break

    boxes = []
    confidences = []
    class_ids = []   # <--- keep class id for each box

    # 6. Parse detections
    for detection in out:
        cx, cy, bw, bh = detection[0:4]
        class_scores = detection[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        # keep any class with decent confidence
        if confidence < 0.5:
            continue

        # convert from 640x640 to original frame coords
        x1 = (cx - bw / 2) * w / 320.0
        y1 = (cy - bh / 2) * h / 320.0
        x2 = (cx + bw / 2) * w / 320.0
        y2 = (cy + bh / 2) * h / 320.0

        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))

        bw_pix = max(0, x2 - x1)
        bh_pix = max(0, y2 - y1)

        if bw_pix == 0 or bh_pix == 0:
            continue

        boxes.append([x1, y1, bw_pix, bh_pix])
        confidences.append(confidence)
        class_ids.append(class_id)

    # 7. NMS
    indices = cv.dnn.NMSBoxes(
        boxes,
        confidences,
        score_threshold=0.5,
        nms_threshold=0.4
    )

    # 8. Draw boxes with class name
    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, bw_pix, bh_pix = boxes[idx]
            cls_id = class_ids[idx]
            conf = confidences[idx]

            # map class id -> human-readable label
            if 0 <= cls_id < len(COCO_CLASSES):
                cls_name = COCO_CLASSES[cls_id]
            else:
                cls_name = f"id_{cls_id}"

            cv.rectangle(frame, (x, y), (x + bw_pix, y + bh_pix), (0, 255, 0), 2)
            label = f"{cls_name} ({conf:.2f})"
            cv.putText(frame, label, (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.imshow("YOLO Detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv.destroyAllWindows()
