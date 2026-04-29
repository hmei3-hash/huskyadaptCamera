"""
find_dist.py — YOLOv8 pose + MiDaS real-world distance estimation
Estimates person distance from camera using YOLO keypoints and
MiDaS inverse-depth with center-frame calibration.

Usage:
    python find_dist.py [--video walking.mp4] [--ref-dist 1.0]
    python find_dist.py --video 0           # webcam

Keys:
    Q - quit
"""

import argparse
import cv2
import numpy as np
import torch


# ── Core functions (pure / easily testable) ─────────────────────

def load_midas(device):
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    return model, transforms.small_transform


@torch.no_grad()
def infer_depth(model, transform, frame, device):
    """Run MiDaS_small; return raw inverse-depth map at original resolution."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).to(device)
    pred = model(inp)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze()
    return pred.cpu().numpy()


def calibrate_scale(depth_raw, ref_dist):
    """
    Compute MiDaS -> meters scale from a known reference distance.
    MiDaS outputs inverse depth: real_dist = scale / raw.
    Uses 30x30 centre-region average as the reference raw value.
    """
    h, w = depth_raw.shape
    cy, cx = h // 2, w // 2
    r = 15
    region = depth_raw[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    raw_val = float(np.mean(region))
    return ref_dist * max(raw_val, 1e-6)


def sample_keypoint_depths(depth_raw, scale, keypoints, conf_thresh=0.5):
    """
    For each visible keypoint in a person, sample the inverse-depth and
    convert to metres.  Skips low-confidence or out-of-bounds keypoints.

    Returns list of (x_px, y_px, dist_m).
    """
    h, w = depth_raw.shape
    samples = []
    for kp in keypoints:
        x, y, conf = kp
        if conf < conf_thresh:
            continue
        xi, yi = int(x), int(y)
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        raw = float(depth_raw[yi, xi])
        dist_m = scale / max(raw, 1e-3)
        samples.append((xi, yi, dist_m))
    return samples


def person_distance(samples):
    """
    Robust distance estimate: median across all visible keypoints.
    Returns None when no samples are available.
    """
    if not samples:
        return None
    return float(np.median([s[2] for s in samples]))


def find_label_anchor(keypoints, conf_thresh=0.5):
    """Return pixel (x, y) of the first visible keypoint for label placement."""
    for kp in keypoints:
        x, y, conf = kp
        if conf >= conf_thresh:
            return int(x), int(y)
    return None


# ── CLI / main loop ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 pose + MiDaS distance")
    p.add_argument("--video", default="walking.mp4",
                   help="Video file path or camera index (default: walking.mp4)")
    p.add_argument("--ref-dist", type=float, default=1.0,
                   help="Known distance (m) to centre of frame at startup")
    return p.parse_args()


def main():
    from ultralytics import YOLO  # heavy import — kept out of module scope for testability

    args = parse_args()

    try:
        source = int(args.video)
    except ValueError:
        source = args.video

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    pose_model = YOLO("yolov8n-pose.pt")
    midas, transform = load_midas(device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source: {source}")
        return

    # Calibrate on first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        cap.release()
        return

    first_depth = infer_depth(midas, transform, first_frame, device)
    scale = calibrate_scale(first_depth, args.ref_dist)
    print(f"[Calibration] scale={scale:.4f}  ref_dist={args.ref_dist}m")

    # Rewind to start for video files
    if not isinstance(source, int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        depth_raw = infer_depth(midas, transform, frame, device)

        results = pose_model(frame, verbose=False)
        for result in results:
            if result.keypoints is None:
                continue

            for person in result.keypoints.data:
                samples = sample_keypoint_depths(depth_raw, scale, person)

                for xi, yi, _ in samples:
                    cv2.circle(frame, (xi, yi), 5, (0, 255, 0), -1)

                dist = person_distance(samples)
                anchor = find_label_anchor(person)
                if dist is not None and anchor is not None:
                    ax, ay = anchor
                    cv2.putText(frame, f"Dist: {dist:.2f}m",
                                (ax, max(ay - 20, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Distance Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
