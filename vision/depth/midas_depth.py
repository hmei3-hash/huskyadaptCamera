"""
MiDaS Monocular Depth Estimation with Calibration
===================================================
Uses MiDaS_small to estimate per-pixel depth from a webcam.
Calibrates against a known reference distance, then displays
real-time depth as a colormap alongside the RGB feed.

Usage:
    python midas_depth.py [--ref-dist 0.5] [--ref-x 320] [--ref-y 300]

Keys:
    Q           - quit
    Left-click  - print real distance at that pixel

"""

import argparse
import cv2
import torch
import numpy as np
import time


def parse_args():
    p = argparse.ArgumentParser(description="MiDaS depth estimation")
    p.add_argument("--ref-dist", type=float, default=0.5,
                   help="Real distance (m) to the calibration reference object")
    p.add_argument("--ref-x", type=int, default=None,
                   help="X pixel of reference object (default: center)")
    p.add_argument("--ref-y", type=int, default=None,
                   help="Y pixel of reference object (default: center)")
    p.add_argument("--input-size", type=int, default=256,
                   help="MiDaS internal resolution (lower = faster)")
    return p.parse_args()


def load_midas(device):
    """Load MiDaS_small model and matching transform."""
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    return model, transforms.small_transform


def calibrate(model, transform, frame, device, input_size, ref_x, ref_y, ref_dist):
    """Compute scale factor from a known reference distance."""
    h, w = frame.shape[:2]
    cx = ref_x if ref_x is not None else w // 2
    cy = ref_y if ref_y is not None else h // 2

    depth = infer_depth(model, transform, frame, device, input_size)
    r = 15
    region = depth[cy - r : cy + r, cx - r : cx + r]
    raw_val = np.mean(region)
    # MiDaS outputs inverse depth: higher raw = closer object.
    # scale = ref_dist * raw_center, so that real_depth = scale / raw
    # gives ref_dist at the calibration point.
    scale = ref_dist * max(raw_val, 1e-6)
    print(f"[Calibration] scale={scale:.6f}  raw_center={raw_val:.2f}")
    return scale


@torch.no_grad()
def infer_depth(model, transform, frame, device, input_size):
    """Run MiDaS inference and return depth map at original resolution."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (input_size, input_size))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).to(device)
    pred = model(inp)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze()
    return pred.cpu().numpy()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MiDaS] Device: {device}")

    model, transform = load_midas(device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        return

    scale = calibrate(model, transform, frame, device,
                      args.input_size, args.ref_x, args.ref_y, args.ref_dist)

    # Mouse callback for point-and-click depth query
    real_depth_map = None

    def on_click(event, x, y, flags, param):
        nonlocal real_depth_map
        if event == cv2.EVENT_LBUTTONDOWN and real_depth_map is not None:
            region = real_depth_map[max(0, y - 10) : y + 10, max(0, x - 10) : x + 10]
            d = np.mean(region)
            print(f"Distance at ({x}, {y}) = {d:.3f} m")

    cv2.namedWindow("RGB | Depth")
    cv2.setMouseCallback("RGB | Depth", on_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        depth = infer_depth(model, transform, frame, device, args.input_size)
        real_depth_map = scale / np.clip(depth, 1e-3, None)

        # Colormap
        d_min, d_max = depth.min(), depth.max()
        norm = ((depth - d_min) / max(d_max - d_min, 1e-6) * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined = np.hstack((frame, colormap))
        cv2.imshow("RGB | Depth", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
