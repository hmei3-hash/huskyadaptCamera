"""
Combined YOLOv8 Object Detection + MiDaS Depth Estimation
=========================================================
Optimizations applied:
  1. YOLO + MiDaS run in parallel via ThreadPoolExecutor
  2. Reduced input resolution (320 for YOLO, 192 for MiDaS)
  3. 8-bit depth processing (skip float64 intermediaries)
  4. Strided depth grid sampling (skip pixels, only sample grid cells)
  5. Skip-frame processing with configurable interval
  6. numpy vectorized detection parsing (no Python for-loop)

Requirements:
  pip install opencv-python numpy torch ultralytics
  Run b.py first to export yolov8n.onnx

Keys:
  Q  - quit
  +/- - increase/decrease skip frames
  Click on window - print depth at that point
"""

import os
import cv2 as cv
import numpy as np
import torch
import time
from concurrent.futures import ThreadPoolExecutor, Future

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════
# CONFIG — tune these for your hardware
# ═══════════════════════════════════════════════════════════════
YOLO_SIZE       = 320        # YOLO input size (320 = fastest)
MIDAS_SIZE      = 192        # MiDaS input size (smaller = faster, default 256)
CONF_THRESH     = 0.45       # YOLO confidence threshold
NMS_THRESH      = 0.45       # NMS IoU threshold
PROCESS_EVERY_N = 2          # process 1 out of N frames (1=all, 2=skip half, etc.)
GRID_ROWS       = 3
GRID_COLS       = 3
CALIB_DISTANCE  = 0.5        # real distance to center at calibration (meters)
CAM_WIDTH       = 640
CAM_HEIGHT      = 480

# ═══════════════════════════════════════════════════════════════
# COCO LABELS
# ═══════════════════════════════════════════════════════════════
COCO_CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
    "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


# ═══════════════════════════════════════════════════════════════
# YOLO INFERENCE (OpenCV DNN, fully vectorized parsing)
# ═══════════════════════════════════════════════════════════════
class YOLODetector:
    def __init__(self, onnx_path, input_size=320, conf=0.45, nms=0.45):
        self.net = cv.dnn.readNetFromONNX(onnx_path)
        self.size = input_size
        self.conf = conf
        self.nms = nms

        # Try CUDA backend — must test with actual forward() to know if it works
        use_cuda = False
        try:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
            # Test inference to verify CUDA actually works
            dummy = np.zeros((1, 3, self.size, self.size), dtype=np.float32)
            self.net.setInput(dummy)
            self.net.forward()
            use_cuda = True
            print("[YOLO] Using CUDA FP16 backend")
        except Exception:
            # Fall back to CPU
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            print("[YOLO] CUDA not available, using CPU backend")

    def detect(self, frame):
        """Returns list of (x1,y1,x2,y2, class_id, confidence)."""
        h, w = frame.shape[:2]
        blob = cv.dnn.blobFromImage(
            frame, 1/255.0, (self.size, self.size), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        out = self.net.forward()[0]  # (84, N) or (N, 84)
        if out.shape[0] < out.shape[1]:
            out = out.T  # -> (N, 84)

        # ── vectorized parsing (no Python for-loop) ──
        scores = out[:, 4:]                           # (N, 80)
        class_ids = np.argmax(scores, axis=1)         # (N,)
        confs = scores[np.arange(len(scores)), class_ids]  # (N,)

        mask = confs >= self.conf
        if not np.any(mask):
            return []

        out   = out[mask]
        confs = confs[mask].astype(np.float32)
        class_ids = class_ids[mask]

        cx, cy, bw, bh = out[:,0], out[:,1], out[:,2], out[:,3]
        sx, sy = w / self.size, h / self.size

        x1 = ((cx - bw/2) * sx).astype(np.int32).clip(0, w-1)
        y1 = ((cy - bh/2) * sy).astype(np.int32).clip(0, h-1)
        x2 = ((cx + bw/2) * sx).astype(np.int32).clip(0, w-1)
        y2 = ((cy + bh/2) * sy).astype(np.int32).clip(0, h-1)

        bw_pix = (x2 - x1).clip(0)
        bh_pix = (y2 - y1).clip(0)
        valid = (bw_pix > 0) & (bh_pix > 0)

        boxes_xywh = np.stack([x1[valid], y1[valid],
                               bw_pix[valid], bh_pix[valid]], axis=1).tolist()
        confs_list = confs[valid].tolist()
        cids_list  = class_ids[valid].tolist()

        indices = cv.dnn.NMSBoxes(boxes_xywh, confs_list,
                                  self.conf, self.nms)
        if len(indices) == 0:
            return []

        results = []
        for idx in indices.flatten():
            bx, by, bbw, bbh = boxes_xywh[idx]
            results.append((bx, by, bx+bbw, by+bbh,
                            int(cids_list[idx]), confs_list[idx]))
        return results


# ═══════════════════════════════════════════════════════════════
# MIDAS DEPTH (reduced resolution + uint8 pipeline)
# ═══════════════════════════════════════════════════════════════
class DepthEstimator:
    def __init__(self, input_size=192, calib_dist=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MiDaS] Device: {self.device}")

        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device).eval()
        # Half precision on GPU
        if self.device.type == "cuda":
            self.model.half()
            print("[MiDaS] Using FP16")

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

        self.input_size = input_size
        self.calib_dist = calib_dist
        self.scale = 1.0
        self.real_depth = None  # latest real-depth map (float32)

    def calibrate(self, frame):
        depth = self._infer(frame)
        h, w = depth.shape
        cy, cx = h//2, w//2
        r = 15
        center_val = np.mean(depth[cy-r:cy+r, cx-r:cx+r])
        self.scale = self.calib_dist / max(center_val, 1e-6)
        print(f"[MiDaS] Calibrated: scale={self.scale:.6f}, center_raw={center_val:.2f}")

    @torch.no_grad()
    def _infer(self, frame):
        h, w = frame.shape[:2]
        small = cv.resize(frame, (self.input_size, self.input_size))
        rgb = cv.cvtColor(small, cv.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        if self.device.type == "cuda":
            inp = inp.half()
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w),
            mode="bilinear", align_corners=False
        ).squeeze()
        return pred.float().cpu().numpy()

    def estimate(self, frame):
        """Run depth, store real_depth, return grid values + colormap."""
        depth = self._infer(frame)
        self.real_depth = depth * self.scale
        return self.real_depth, self._colormap_uint8(depth)

    @staticmethod
    def _colormap_uint8(depth):
        """8-bit normalized colormap — no float64 intermediate."""
        dmin, dmax = depth.min(), depth.max()
        rng = max(dmax - dmin, 1e-6)
        norm = ((depth - dmin) * (255.0 / rng)).astype(np.uint8)
        return cv.applyColorMap(norm, cv.COLORMAP_MAGMA)


# ═══════════════════════════════════════════════════════════════
# GRID OVERLAY (strided sampling — skips most pixels)
# ═══════════════════════════════════════════════════════════════
COLOR_NEAR = np.array([200, 60,  20], dtype=np.float32)
COLOR_FAR  = np.array([20,  40, 200], dtype=np.float32)
FONT       = cv.FONT_HERSHEY_DUPLEX

GRID_SAMPLE_STRIDE = 4


def draw_grid_overlay(frame, real_depth):
    h, w = frame.shape[:2]
    out = frame.copy()
    ch, cw = h // GRID_ROWS, w // GRID_COLS

    vals = np.empty((GRID_ROWS, GRID_COLS), dtype=np.float32)
    for r in range(GRID_ROWS):
        y0 = r * ch
        y1 = (y0 + ch) if r < GRID_ROWS - 1 else h
        for c in range(GRID_COLS):
            x0 = c * cw
            x1 = (x0 + cw) if c < GRID_COLS - 1 else w
            vals[r, c] = np.mean(real_depth[y0:y1:GRID_SAMPLE_STRIDE,
                                             x0:x1:GRID_SAMPLE_STRIDE])

    vmin, vmax = vals.min(), vals.max()
    vrange = max(vmax - vmin, 1e-6)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val = vals[r, c]
            t = (val - vmin) / vrange
            y0 = r * ch;  y1 = (y0 + ch) if r < GRID_ROWS - 1 else h
            x0 = c * cw;  x1 = (x0 + cw) if c < GRID_COLS - 1 else w
            mcx, mcy = (x0+x1)//2, (y0+y1)//2

            color = tuple(int(x) for x in (1-t)*COLOR_NEAR + t*COLOR_FAR)
            ov = out.copy()
            cv.rectangle(ov, (x0,y0), (x1,y1), color, -1)
            cv.addWeighted(ov, 0.25, out, 0.75, 0, out)

            is_ctr = (r == 1 and c == 1)
            cv.rectangle(out, (x0,y0), (x1,y1),
                         (255,255,255) if is_ctr else (140,140,140),
                         3 if is_ctr else 1)

            label = f"{val:.2f}m"
            (tw, th_), _ = cv.getTextSize(label, FONT, 0.5, 1)
            tx, ty = mcx - tw//2, mcy + th_//2
            cv.rectangle(out, (tx-2, ty-th_-2), (tx+tw+2, ty+4), (0,0,0), -1)
            cv.putText(out, label, (tx, ty), FONT, 0.5, (255,255,255), 1, cv.LINE_AA)

            if is_ctr:
                (tw2, th2), _ = cv.getTextSize("REF", FONT, 0.35, 1)
                cv.putText(out, "REF", (mcx-tw2//2, mcy-th_//2-6),
                           FONT, 0.35, (0,255,255), 1, cv.LINE_AA)

    bar = np.zeros((28, w, 3), dtype=np.uint8)
    ctr_val = vals[1,1]
    txt = (f"MIN {vmin:.2f}m  MAX {vmax:.2f}m  "
           f"CTR {ctr_val:.2f}m  RNG {vmax-vmin:.2f}m")
    cv.putText(bar, txt, (6,20), FONT, 0.38, (160,230,255), 1, cv.LINE_AA)
    return np.vstack([bar, out])


# ═══════════════════════════════════════════════════════════════
# DRAW YOLO BOXES ON FRAME
# ═══════════════════════════════════════════════════════════════
def draw_detections(frame, detections, depth_map=None):
    for (x1, y1, x2, y2, cls_id, conf) in detections:
        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"id_{cls_id}"
        label = f"{name} {conf:.0%}"

        if depth_map is not None:
            cy_ = (y1 + y2) // 2
            cx_ = (x1 + x2) // 2
            h, w = depth_map.shape
            if 0 <= cy_ < h and 0 <= cx_ < w:
                r = 5
                region = depth_map[max(0,cy_-r):cy_+r, max(0,cx_-r):cx_+r]
                if region.size > 0:
                    d = float(np.mean(region))
                    label += f" {d:.1f}m"

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th_), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(frame, (x1, y1-th_-6), (x1+tw+4, y1), (0,255,0), -1)
        cv.putText(frame, label, (x1+2, y1-4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)


# ═══════════════════════════════════════════════════════════════
# MOUSE CALLBACK
# ═══════════════════════════════════════════════════════════════
depth_estimator_global = None

def mouse_cb(event, x, y, flags, param):
    if event != cv.EVENT_LBUTTONDOWN:
        return
    if depth_estimator_global is None or depth_estimator_global.real_depth is None:
        return
    ry = y - 28
    rd = depth_estimator_global.real_depth
    if 0 <= ry < rd.shape[0] and 0 <= x < rd.shape[1]:
        region = rd[max(0,ry-10):ry+10, max(0,x-10):x+10]
        print(f"Click ({x},{ry}) -> {np.mean(region):.3f} m")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    global depth_estimator_global

    print("=" * 55)
    print("  YOLO + MiDaS  |  Optimized Combined Pipeline")
    print("=" * 55)

    # Init YOLO
    onnx_file = os.path.join(SCRIPT_DIR, "yolov8n.onnx")
    if not os.path.isfile(onnx_file):
        print(f"ERROR: Cannot find {onnx_file}")
        print("Run b.py first to export the ONNX model.")
        return
    yolo = YOLODetector(onnx_file, YOLO_SIZE, CONF_THRESH, NMS_THRESH)

    # Init MiDaS
    depth = DepthEstimator(MIDAS_SIZE, CALIB_DISTANCE)
    depth_estimator_global = depth

    # ── Open camera (same pattern as working a.py / test3.py) ──
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit()

    # Read first frame for calibration
    ret, first = cap.read()
    if not ret:
        print("Error: Cannot read from camera")
        cap.release()
        return

    print(f"[Camera] Opened OK  frame={first.shape[1]}x{first.shape[0]}")
    depth.calibrate(first)

    # Thread pool for parallel YOLO + MiDaS
    pool = ThreadPoolExecutor(max_workers=2)

    cv.namedWindow("YOLO + Depth")
    cv.setMouseCallback("YOLO + Depth", mouse_cb)

    frame_idx = 0
    process_n = PROCESS_EVERY_N

    # Cached results (reused on skipped frames)
    last_detections = []
    last_real_depth = None
    last_colormap   = None

    fps_alpha = 0.9
    fps_smooth = 0.0

    print(f"[Config] skip={process_n-1} of {process_n} frames | "
          f"YOLO@{YOLO_SIZE} | MiDaS@{MIDAS_SIZE}")
    print("Press Q to quit, +/- to adjust frame skip\n")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        do_process = (frame_idx % process_n == 0)

        if do_process:
            # ── Run YOLO and MiDaS in PARALLEL ──
            fut_yolo:  Future = pool.submit(yolo.detect, frame)
            fut_depth: Future = pool.submit(depth.estimate, frame)

            last_detections = fut_yolo.result()
            last_real_depth, last_colormap = fut_depth.result()

        # ── Draw ──
        display = frame.copy()

        if last_real_depth is not None:
            display = draw_grid_overlay(display, last_real_depth)
        else:
            bar = np.zeros((28, display.shape[1], 3), dtype=np.uint8)
            display = np.vstack([bar, display])

        draw_detections(display, last_detections,
                        last_real_depth if last_real_depth is not None else None)

        # right panel: depth colormap
        if last_colormap is not None:
            h_disp = display.shape[0]
            h_cm   = last_colormap.shape[0]
            diff = h_disp - h_cm
            if diff > 0:
                bar_r = np.zeros((diff, last_colormap.shape[1], 3), dtype=np.uint8)
                cv.putText(bar_r, "DEPTH", (6,min(20,diff-2)),
                           FONT, 0.38, (180,180,180), 1, cv.LINE_AA)
                right = np.vstack([bar_r, last_colormap])
            else:
                right = last_colormap[:h_disp]
            combined = np.hstack([display, right])
        else:
            combined = display

        # FPS
        dt = max(time.time() - t0, 1e-6)
        fps_now = 1.0 / dt
        fps_smooth = fps_alpha * fps_smooth + (1 - fps_alpha) * fps_now
        cv.putText(combined, f"FPS {fps_smooth:.0f}  skip={process_n}",
                   (8, combined.shape[0] - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1, cv.LINE_AA)

        cv.imshow("YOLO + Depth", combined)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            process_n = min(process_n + 1, 10)
            print(f"[Config] process every {process_n} frames")
        elif key == ord('-') or key == ord('_'):
            process_n = max(process_n - 1, 1)
            print(f"[Config] process every {process_n} frames")

    cap.release()
    cv.destroyAllWindows()
    pool.shutdown(wait=False)
    print("Done.")


if __name__ == "__main__":
    main()