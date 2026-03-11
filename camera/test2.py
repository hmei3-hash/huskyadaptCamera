"""
实时深度网格 - 摄像头版
用法: python depth_grid_live.py <中心格距离(米)>
例如: python depth_grid_live.py 0.50

按键:
  Q / ESC  退出
  空格      立即触发一次分析
  +/-       调整刷新间隔
"""

import cv2
import numpy as np
import base64, json, threading, time, sys
import anthropic

# ─── 配置 ────────────────────────────────────────────────────────────────────
CAMERA_ID      = 0
FRAME_W        = 640
FRAME_H        = 480
GRID_ROWS      = 3
GRID_COLS      = 3
CENTER_CELL    = (1, 1)
API_INTERVAL   = 3.0          # 秒：两次 API 调用之间的最小间隔
ALPHA_OVERLAY  = 0.35         # 蒙版透明度
SMOOTH_FACTOR  = 0.25         # 网格平滑系数（EMA），0=不平滑, 1=无平滑

COLOR_NEAR     = np.array([200, 60,  20], dtype=float)   # BGR 蓝
COLOR_FAR      = np.array([20,  40, 200], dtype=float)   # BGR 红
FONT           = cv2.FONT_HERSHEY_DUPLEX
# ─────────────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic()

PROMPT_TMPL = """You are a depth estimation expert.

Image divided into 3×3 equal cells:
  [0,0]=top-left  [0,1]=top-center  [0,2]=top-right
  [1,0]=mid-left  [1,1]=CENTER      [1,2]=mid-right
  [2,0]=bot-left  [2,1]=bot-center  [2,2]=bot-right

CENTER cell [1,1] real distance = {dist:.4f} meters.

Estimate depth for ALL 9 cells scaled so center = {dist:.4f} m.
Return ONLY JSON, no markdown:
{{"grid":[[v00,v01,v02],[v10,v11,v12],[v20,v21,v22]]}}
Values: floats in meters, 2 decimal places."""


def lerp_color(t: float):
    c = (1 - t) * COLOR_NEAR + t * COLOR_FAR
    return tuple(int(x) for x in c)


def put_text_centered(img, text, cx, cy, scale, thick, color, bg=None):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = cx - tw // 2, cy + th // 2
    if bg is not None:
        p = 4
        cv2.rectangle(img, (x-p, y-th-p), (x+tw+p, y+bl+p), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_grid(frame: np.ndarray, grid: list, alpha: float = 1.0) -> np.ndarray:
    """把 3×3 深度网格叠加到帧上，alpha 控制整体不透明度（淡入用）"""
    h, w = frame.shape[:2]
    out = frame.copy()

    flat  = [v for row in grid for v in row]
    vmin, vmax = min(flat), max(flat)
    vrange = max(vmax - vmin, 1e-6)

    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            val = grid[r][c]
            t   = (val - vmin) / vrange
            x0, y0 = c * cell_w, r * cell_h
            x1 = x0 + cell_w if c < GRID_COLS - 1 else w
            y1 = y0 + cell_h if r < GRID_ROWS - 1 else h
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

            # 半透明色块
            ov = out.copy()
            cv2.rectangle(ov, (x0, y0), (x1, y1), lerp_color(t), -1)
            cv2.addWeighted(ov, ALPHA_OVERLAY * alpha, out, 1 - ALPHA_OVERLAY * alpha, 0, out)

            # 边框
            is_center = (r, c) == CENTER_CELL
            cv2.rectangle(out, (x0, y0), (x1, y1),
                           (255, 255, 255) if is_center else (160, 160, 160),
                           3 if is_center else 1)

            # 距离文字
            put_text_centered(out, f"{val:.2f}m", cx, cy, 0.60, 1, (255,255,255), (0,0,0))
            if is_center:
                put_text_centered(out, "REF", cx, cy - 24, 0.42, 1, (0,255,255), (0,0,0))

    # 顶部状态栏
    bar = np.zeros((34, w, 3), dtype=np.uint8)
    txt = (f"MIN {vmin:.2f}m  MAX {vmax:.2f}m  "
           f"CENTER {grid[CENTER_CELL[0]][CENTER_CELL[1]]:.2f}m  "
           f"RANGE {vmax-vmin:.2f}m")
    cv2.putText(bar, txt, (8, 23), FONT, 0.46, (160, 230, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


class DepthAnalyzer:
    """后台线程：定时调用 API，主线程随时读取最新网格"""

    def __init__(self, center_dist: float):
        self.center_dist   = center_dist
        self.grid          = None          # 最新网格（平滑后）
        self.raw_grid      = None          # 最新 API 返回
        self.lock          = threading.Lock()
        self.busy          = False
        self.last_call     = 0.0
        self.interval      = API_INTERVAL
        self.status        = "等待首帧..."
        self._thread       = None

    def request(self, frame: np.ndarray):
        """提交一帧，若条件满足则在后台线程发起 API 调用"""
        now = time.time()
        if self.busy or (now - self.last_call) < self.interval:
            return
        self.busy = True
        self.last_call = now
        self.status = "分析中..."
        img = frame.copy()
        t = threading.Thread(target=self._call_api, args=(img,), daemon=True)
        t.start()

    def _call_api(self, frame: np.ndarray):
        try:
            # 压缩到 512px 宽，减少 token
            scale = 512 / frame.shape[1]
            small = cv2.resize(frame, (512, int(frame.shape[0] * scale)))
            _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.standard_b64encode(buf.tobytes()).decode()

            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text",
                         "text": PROMPT_TMPL.format(dist=self.center_dist)}
                    ]
                }]
            )
            raw  = msg.content[0].text.strip().replace("```json","").replace("```","")
            new_grid = json.loads(raw)["grid"]

            with self.lock:
                if self.grid is None:
                    self.grid = new_grid
                else:
                    # EMA 平滑
                    for r in range(GRID_ROWS):
                        for c in range(GRID_COLS):
                            self.grid[r][c] = (SMOOTH_FACTOR * new_grid[r][c]
                                               + (1 - SMOOTH_FACTOR) * self.grid[r][c])
                self.raw_grid = new_grid
                self.status   = f"更新于 {time.strftime('%H:%M:%S')}"
        except Exception as e:
            with self.lock:
                self.status = f"错误: {e}"
        finally:
            self.busy = False

    def get(self):
        with self.lock:
            return self.grid, self.status


def main():
    if len(sys.argv) < 2:
        print("用法: python depth_grid_live.py <中心格距离(米)>")
        print("例如: python depth_grid_live.py 0.50")
        sys.exit(1)

    center_dist = float(sys.argv[1])
    print(f"摄像头 {CAMERA_ID} 已开启  中心参考距离: {center_dist} m")
    print("按 Q/ESC 退出  空格=立即分析  +/-=调节间隔")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("无法打开摄像头")
        sys.exit(1)

    analyzer  = DepthAnalyzer(center_dist)
    fade_t    = 0.0          # 淡入起始时间
    prev_grid = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            break

        # 提交当前帧给后台分析
        analyzer.request(frame)

        grid, status = analyzer.get()
        display = frame.copy()

        if grid is not None:
            # 新网格到来时重置淡入
            if grid is not prev_grid:
                fade_t    = time.time()
                prev_grid = grid
            alpha = min(1.0, (time.time() - fade_t) / 0.4)   # 0.4s 淡入
            display = draw_grid(frame, grid, alpha)
        else:
            # 还没有网格：只画等待提示
            h, w = frame.shape[:2]
            bar = np.zeros((34, w, 3), dtype=np.uint8)
            cv2.putText(bar, status, (8, 23), FONT, 0.48, (100, 200, 255), 1, cv2.LINE_AA)
            display = np.vstack([bar, frame])

        # 底部调试信息
        h2, w2 = display.shape[:2]
        bz = f"间隔 {analyzer.interval:.1f}s  {'分析中...' if analyzer.busy else status}"
        cv2.putText(display, bz, (8, h2 - 8), FONT, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.imshow("Realtime Depth Grid  [Q=quit  Space=now  +/-=interval]", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == ord(' '):
            # 强制立即分析
            analyzer.last_call = 0
        elif key == ord('+') or key == ord('='):
            analyzer.interval = max(1.0, analyzer.interval - 0.5)
            print(f"间隔 → {analyzer.interval:.1f}s")
        elif key == ord('-'):
            analyzer.interval = min(30.0, analyzer.interval + 0.5)
            print(f"间隔 → {analyzer.interval:.1f}s")

    cap.release()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()