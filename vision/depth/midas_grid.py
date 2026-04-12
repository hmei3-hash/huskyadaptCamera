"""
MiDaS Depth with 3×3 Grid Overlay
===================================
Extends midas_depth.py with a 3×3 spatial grid showing per-cell
average distance, color-coded from near (blue) to far (red).

Usage:
    python midas_grid.py [--ref-dist 0.5] [--grid 3x3]

Keys:
    Q           - quit
    Left-click  - print real distance at that pixel
"""

import argparse
import cv2
import torch
import numpy as np
import time

# Reuse core functions from the depth module
from midas_depth import load_midas, infer_depth, calibrate


# ── Grid drawing ──────────────────────────────────────────────

COLOR_NEAR = np.array([200, 60, 20], dtype=float)   # BGR blue (near)
COLOR_FAR  = np.array([20, 40, 200], dtype=float)    # BGR red  (far)
FONT = cv2.FONT_HERSHEY_DUPLEX


def lerp_color(t):
    c = (1 - t) * COLOR_NEAR + t * COLOR_FAR
    return tuple(int(x) for x in c)


def put_text_centered(img, text, cx, cy, scale, thick, color, bg=None):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = cx - tw // 2, cy + th // 2
    if bg is not None:
        p = 3
        cv2.rectangle(img, (x - p, y - th - p), (x + tw + p, y + bl + p), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_grid_overlay(frame, real_depth, rows=3, cols=3):
    """Overlay a rows×cols grid with per-cell average distance."""
    h, w = frame.shape[:2]
    out = frame.copy()
    cell_h, cell_w = h // rows, w // cols

    # Compute per-cell averages
    vals = np.empty((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h if r < rows - 1 else h
            x0, x1 = c * cell_w, (c + 1) * cell_w if c < cols - 1 else w
            vals[r, c] = np.mean(real_depth[y0:y1, x0:x1])

    vmin, vmax = vals.min(), vals.max()
    vrange = max(vmax - vmin, 1e-6)

    for r in range(rows):
        for c in range(cols):
            val = vals[r, c]
            t = (val - vmin) / vrange

            y0 = r * cell_h
            y1 = (r + 1) * cell_h if r < rows - 1 else h
            x0 = c * cell_w
            x1 = (c + 1) * cell_w if c < cols - 1 else w
            mcx, mcy = (x0 + x1) // 2, (y0 + y1) // 2

            # Semi-transparent color block
            ov = out.copy()
            cv2.rectangle(ov, (x0, y0), (x1, y1), lerp_color(t), -1)
            cv2.addWeighted(ov, 0.30, out, 0.70, 0, out)

            # Border
            is_center = (r == rows // 2 and c == cols // 2)
            cv2.rectangle(out, (x0, y0), (x1, y1),
                          (255, 255, 255) if is_center else (160, 160, 160),
                          3 if is_center else 1)

            # Distance label
            put_text_centered(out, f"{val:.2f}m", mcx, mcy,
                              0.58, 1, (255, 255, 255), (0, 0, 0))
            if is_center:
                put_text_centered(out, "REF", mcx, mcy - 22,
                                  0.40, 1, (0, 255, 255), (0, 0, 0))

    # Top status bar
    bar = np.zeros((32, w, 3), dtype=np.uint8)
    ctr = vals[rows // 2, cols // 2]
    txt = f"MIN {vmin:.2f}m  MAX {vmax:.2f}m  CENTER {ctr:.2f}m  RANGE {vmax - vmin:.2f}m"
    cv2.putText(bar, txt, (8, 22), FONT, 0.44, (160, 230, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


# ── Main ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MiDaS depth with grid overlay")
    p.add_argument("--ref-dist", type=float, default=0.5)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--grid", default="3x3", help="Grid layout, e.g. 3x3 or 4x4")
    return p.parse_args()


def main():
    args = parse_args()
    rows, cols = (int(x) for x in args.grid.split("x"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_midas(device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        return

    scale = calibrate(model, transform, frame, device,
                      args.input_size, None, None, args.ref_dist)

    real_depth_map = None

    def on_click(event, x, y, flags, param):
        nonlocal real_depth_map
        if event == cv2.EVENT_LBUTTONDOWN and real_depth_map is not None:
            ry = y - 32  # offset for status bar
            if 0 <= ry < real_depth_map.shape[0] and 0 <= x < real_depth_map.shape[1]:
                region = real_depth_map[max(0, ry - 10) : ry + 10,
                                        max(0, x - 10) : x + 10]
                print(f"Distance at ({x}, {ry}) = {np.mean(region):.3f} m")

    cv2.namedWindow("RGB | Depth")
    cv2.setMouseCallback("RGB | Depth", on_click)

    print("Running... Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        depth = infer_depth(model, transform, frame, device, args.input_size)
        real_depth_map = depth * scale

        # Left panel: RGB + grid
        left = draw_grid_overlay(frame, real_depth_map, rows, cols)

        # Right panel: depth colormap
        d_min, d_max = depth.min(), depth.max()
        norm = ((depth - d_min) / max(d_max - d_min, 1e-6) * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)

        bar_r = np.zeros((32, frame.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar_r, "DEPTH MAP", (8, 22), FONT, 0.44,
                    (200, 200, 200), 1, cv2.LINE_AA)
        right = np.vstack([bar_r, colormap])

        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(left, f"FPS {fps:.1f}", (8, left.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1, cv2.LINE_AA)

        cv2.imshow("RGB | Depth", np.hstack((left, right)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
