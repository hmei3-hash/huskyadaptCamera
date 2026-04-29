from ultralytics import YOLO
import torch
import cv2
import numpy as np

model = YOLO("yolov8n-pose.pt")

# --- MiDaS setup ---
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.eval()
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

cap = cv2.VideoCapture('walking.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- MiDaS depth map ---
    imgbatch = transform(frame).to('cpu')
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, 
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # --- YOLO pose ---
    results = model(frame)

    for result in results:
        if result.keypoints is None:
            continue

        for person in result.keypoints.data:  # loop each detected person
            # person shape: [17, 3] — 17 keypoints, each (x, y, conf)
            visible_depths = []

            for kp in person:
                x, y, conf = kp
                if conf < 0.5:  # skip low confidence / not visible
                    continue

                x, y = int(x), int(y)

                # Sample MiDaS depth at this keypoint's pixel location
                depth_val = depth_map[y, x]
                visible_depths.append(depth_val)

                # Draw on frame
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            if visible_depths:
                # Median across all visible keypoints = stable depth estimate
                person_depth = np.median(visible_depths)
                cv2.putText(frame, f"Depth: {person_depth:.1f}", 
                           (int(person[0][0]), int(person[0][1]) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Walking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()