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

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# %%
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

small_size = 256

# %%
# -------------------------
# Calibration
# 用图像中心区域对应的真实距离来标定 scale
# -------------------------
REAL_DISTANCE = 0.5   # ← 默认中心格真实距离（米），按需修改

print("Calibrating...")
ret, frame = cap.read()
if not ret:
    print("Camera error")
    exit()

original_h, original_w = frame.shape[:2]

img_small = cv2.resize(frame, (small_size, small_size))
img_rgb   = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
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

# 用图像中心 30×30 区域均值作为标定基准
cy, cx = original_h // 2, original_w // 2
r = 15
center_raw = np.mean(depth[cy-r:cy+r, cx-r:cx+r])
scale = REAL_DISTANCE / center_raw
print(f"Scale calibrated: {scale:.6f}  (center raw={center_raw:.2f})")

# %%
# -------------------------
# 3×3 网格绘制
# -------------------------
GRID_ROWS = 3
GRID_COLS = 3
COLOR_NEAR = np.array([200, 60,  20], dtype=float)   # BGR 蓝（近）
COLOR_FAR  = np.array([20,  40, 200], dtype=float)   # BGR 红（远）
FONT = cv2.FONT_HERSHEY_DUPLEX


def lerp_color(t):
    c = (1 - t) * COLOR_NEAR + t * COLOR_FAR
    return tuple(int(x) for x in c)


def put_text_centered(img, text, cx, cy, scale, thick, color, bg=None):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = cx - tw // 2, cy + th // 2
    if bg is not None:
        p = 3
        cv2.rectangle(img, (x-p, y-th-p), (x+tw+p, y+bl+p), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_grid_overlay(frame, real_depth_map):
    """
    将图像切成 3×3，每格取所有像素均值距离，
    叠加半透明色块 + 距离文字，返回叠加后图像。
    """
    h, w = frame.shape[:2]
    out  = frame.copy()

    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS

    # 先收集 9 格均值，用于归一化颜色
    grid_vals = []
    for r in range(GRID_ROWS):
        row_vals = []
        for c in range(GRID_COLS):
            y0 = r * cell_h
            y1 = y0 + cell_h if r < GRID_ROWS - 1 else h
            x0 = c * cell_w
            x1 = x0 + cell_w if c < GRID_COLS - 1 else w
            avg = float(np.mean(real_depth_map[y0:y1, x0:x1]))
            row_vals.append(avg)
        grid_vals.append(row_vals)

    flat  = [v for row in grid_vals for v in row]
    vmin, vmax = min(flat), max(flat)
    vrange = max(vmax - vmin, 1e-6)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val = grid_vals[r][c]
            t   = (val - vmin) / vrange

            y0 = r * cell_h
            y1 = y0 + cell_h if r < GRID_ROWS - 1 else h
            x0 = c * cell_w
            x1 = x0 + cell_w if c < GRID_COLS - 1 else w
            mcx, mcy = (x0 + x1) // 2, (y0 + y1) // 2

            # 半透明色块
            ov = out.copy()
            cv2.rectangle(ov, (x0, y0), (x1, y1), lerp_color(t), -1)
            cv2.addWeighted(ov, 0.30, out, 0.70, 0, out)

            # 边框
            is_center = (r == 1 and c == 1)
            cv2.rectangle(out, (x0, y0), (x1, y1),
                          (255, 255, 255) if is_center else (160, 160, 160),
                          3 if is_center else 1)

            # 距离文字
            put_text_centered(out, f"{val:.2f}m", mcx, mcy,
                               0.58, 1, (255, 255, 255), (0, 0, 0))
            if is_center:
                put_text_centered(out, "REF", mcx, mcy - 22,
                                   0.40, 1, (0, 255, 255), (0, 0, 0))

    # 顶部统计栏
    bar = np.zeros((32, w, 3), dtype=np.uint8)
    center_val = grid_vals[1][1]
    txt = (f"MIN {vmin:.2f}m  MAX {vmax:.2f}m  "
           f"CENTER {center_val:.2f}m  RANGE {vmax-vmin:.2f}m")
    cv2.putText(bar, txt, (8, 22), FONT, 0.44, (160, 230, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


# %%
# -------------------------
# 鼠标点击显示真实距离
# -------------------------
real_depth_map = None

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and real_depth_map is not None:
        # 点击坐标要减去顶部状态栏高度（32px）
        ry = y - 32
        if 0 <= ry < real_depth_map.shape[0] and 0 <= x < real_depth_map.shape[1]:
            region = real_depth_map[max(0,ry-10):ry+10, max(0,x-10):x+10]
            d = float(np.mean(region))
            print(f"点击 ({x}, {ry})  真实距离 = {d:.3f} m")

cv2.namedWindow("RGB | Depth")
cv2.setMouseCallback("RGB | Depth", mouse_callback)

# %%
# -------------------------
# 主循环
# -------------------------
print("运行中... 按 Q 退出")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    original_h, original_w = frame.shape[:2]

    img_small   = cv2.resize(frame, (small_size, small_size))
    img_rgb     = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
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
    real_depth_map = depth * scale   # 全图真实距离（米）

    # ── 左：RGB + 3×3 网格叠加 ──
    left = draw_grid_overlay(frame, real_depth_map)

    # ── 右：深度伪彩色图 ──
    depth_norm     = (depth - depth.min()) / (depth.max() - depth.min())
    depth_colormap = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA
    )
    # 右图顶部加同高空白栏，保持与左图对齐
    bar_r = np.zeros((32, original_w, 3), dtype=np.uint8)
    cv2.putText(bar_r, "DEPTH MAP", (8, 22), FONT, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    right = np.vstack([bar_r, depth_colormap])

    # FPS 叠加到左图
    fps = 1.0 / max(time.time() - t0, 1e-6)
    cv2.putText(left, f"FPS {fps:.1f}", (8, left.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1, cv2.LINE_AA)

    combined = np.hstack((left, right))
    cv2.imshow("RGB | Depth", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()