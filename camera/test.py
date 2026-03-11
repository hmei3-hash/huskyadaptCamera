# %%
import cv2
import torch
import numpy as np
import time


# %%

# -------------------------
# Load MiDaS model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_type = "MiDaS_small"   # 用小模型更快
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# %%
cap = cv2.VideoCapture(0)

# 想更快可以降低摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# %%
print("Calibrating...")

ret, frame = cap.read()
if not ret:
    print("Camera error")
    exit()

original_h, original_w = frame.shape[:2]

small_size = 256
img_small = cv2.resize(frame, (small_size, small_size))
img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(original_h, original_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth = prediction.cpu().numpy()

# ========= 你需要改这里 =========
REAL_DISTANCE = 0.03     # 圆柱真实距离（米）
CYL_X, CYL_Y = 320, 300  # 圆柱中心像素坐标
# =================================

cyl_region = depth[CYL_Y-15:CYL_Y+15, CYL_X-15:CYL_X+15]
cyl_depth = np.mean(cyl_region)

scale = REAL_DISTANCE / cyl_depth
print("Scale calibrated:", scale)


# %%
# -----------------------------
# 4️⃣ 鼠标点击显示真实距离
# -----------------------------
real_depth_map = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if real_depth_map is not None:
            region = real_depth_map[y-10:y+10, x-10:x+10]
            d = np.mean(region)
            print(f"Real distance at ({x},{y}) = {d:.3f} meters")

cv2.namedWindow("RGB | Depth")
cv2.setMouseCallback("RGB | Depth", mouse_callback)

# -----------------------------
# 5️⃣ 主循环
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    original_h, original_w = frame.shape[:2]

    img_small = cv2.resize(frame, (small_size, small_size))
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(original_h, original_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # 转换为真实距离
    real_depth_map = depth * scale

    # 归一化显示
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_colormap = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8),
        cv2.COLORMAP_MAGMA
    )

    # FPS
    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    combined = np.hstack((frame, depth_colormap))

    cv2.imshow("RGB | Depth", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# %%


# %%



