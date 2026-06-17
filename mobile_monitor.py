"""
手机本地监控 - YOLOv8实时目标检测

功能:
- 调用手机摄像头进行实时视频采集
- 使用.pt模型在本地进行推理
- 连续3-5帧检测到人玩手机行为时触发告警
- 将抓拍帧转为Base64编码，通过局域网POST发送至FastAPI后端

Pydroid 3 依赖安装:
    pip install opencv-python numpy requests ultralytics pillow
"""

import base64
import json
import sqlite3
import time
from typing import List

import cv2
import numpy as np
import requests
import tkinter as tk
from PIL import Image, ImageTk
import torch
import torch.nn as nn

# ============================================================
# 自定义模块定义 (必须与 ultralytics 包中完全一致)
# ============================================================

class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions."""

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv, DWConv
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class SimSPPF(nn.Module):
    """Simplified Spatial Pyramid Pooling - Fast (SimSPPF) layer."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=True):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class C2fLightConv(nn.Module):
    """C2f module with LightConv instead of standard Bottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv, LightConv
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(LightConv(self.c, self.c, k=3) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""

    def __init__(self, c1, r=16):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // r, bias=False), nn.ReLU(inplace=True), nn.Linear(c1 // r, c1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv(2, 1, k=7, p=3, act=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(spatial_att))
        x = x * spatial_att
        return x


class C2fCBAM(nn.Module):
    """C2f module with CBAM attention."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.block import Bottleneck
        self.c = max(int(c2 * e), 1)
        c2 = max(c2, 1)
        nn.Module.__init__(self)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(max((2 + n) * self.c, 1), c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cbam = CBAM(self.c)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(self.cbam(m(y[-1])) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# 注册自定义模块到ultralytics (必须在导入YOLO之前)
import sys
# 确保 ultralytics 模块已加载
import ultralytics.nn.modules.block as _block
import ultralytics.nn.tasks as _tasks

# 将自定义类注入到 ultralytics 的 block 和 tasks 模块中
setattr(_block, 'C2fLightConv', C2fLightConv)
setattr(_block, 'SimSPPF', SimSPPF)
setattr(_block, 'C2fCBAM', C2fCBAM)
setattr(_block, 'LightConv', LightConv)
setattr(_block, 'CBAM', CBAM)

setattr(_tasks, 'C2fLightConv', C2fLightConv)
setattr(_tasks, 'SimSPPF', SimSPPF)
setattr(_tasks, 'C2fCBAM', C2fCBAM)
setattr(_tasks, 'LightConv', LightConv)
setattr(_tasks, 'CBAM', CBAM)

# 同时注入到 sys.modules 中，确保 pickle 能找到
sys.modules['ultralytics.nn.modules.block'].C2fLightConv = C2fLightConv
sys.modules['ultralytics.nn.modules.block'].SimSPPF = SimSPPF
sys.modules['ultralytics.nn.modules.block'].C2fCBAM = C2fCBAM
sys.modules['ultralytics.nn.modules.block'].LightConv = LightConv
sys.modules['ultralytics.nn.modules.block'].CBAM = CBAM

sys.modules['ultralytics.nn.tasks'].C2fLightConv = C2fLightConv
sys.modules['ultralytics.nn.tasks'].SimSPPF = SimSPPF
sys.modules['ultralytics.nn.tasks'].C2fCBAM = C2fCBAM
sys.modules['ultralytics.nn.tasks'].LightConv = LightConv
sys.modules['ultralytics.nn.tasks'].CBAM = CBAM

from ultralytics import YOLO


# ============================================================
# 配置参数
# ============================================================

# 模型路径 (改为.pt)
MODEL_PATH = "/storage/emulated/0/Download/best.pt"

# 告警后端API地址 (修改为电脑的局域网IP)
SERVER_URL = "http://192.168.241.61:8001"
ALERT_ENDPOINT = f"{SERVER_URL}/api/alert"

# 检测阈值 (进一步降低以提高手机检测率)
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45

# 连续帧触发告警的阈值
CONSECUTIVE_FRAMES_THRESHOLD = 3

# 摄像头索引 (0=后置, 1=前置)
CAMERA_INDEX = 0

# 跳帧检测 (每N帧推理一次, 提升FPS)
SKIP_FRAMES = 2

# 模型输入尺寸 (增大以提高检测精度)
MODEL_IMGSZ = 640


# ============================================================
# YOLODetector - YOLO推理引擎
# ============================================================

class YOLODetector:
    """YOLOv8推理引擎"""

    def __init__(self, model_path: str, conf_threshold: float = 0.4, iou_threshold: float = 0.45, imgsz: int = 320):
        """
        初始化检测器

        Args:
            model_path: 模型文件路径 (.pt)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU阈值
            imgsz: 模型输入尺寸 (越小越快)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        # 加载模型
        self.model = YOLO(model_path, task="detect")
        print(f"模型加载成功: {model_path}")

        # 动态获取模型类别名称
        self.class_names = self.model.names
        print(f"模型类别: {self.class_names}")

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        对单帧图像进行目标检测

        Args:
            frame: 输入图像 (numpy array, BGR格式)

        Returns:
            检测结果列表
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                detections.append({
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                    "confidence": confidence,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

        return detections


# ============================================================
# AlertManager - 告警管理器
# ============================================================

class AlertManager:
    """告警管理器 - 连续帧检测与告警上传 + 本地SQLite存储"""

    def __init__(self, threshold: int = 3, server_url: str = "http://127.0.0.1:8001", class_names: dict = None, db_path: str = "alerts.db"):
        self.threshold = threshold
        self.server_url = server_url
        self.consecutive_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 10  # 告警冷却时间(秒)
        self.class_names = class_names or {}
        self.db_path = db_path
        # 查找目标类别名称 (手机)
        self.target_name = None
        for cid, name in self.class_names.items():
            if "phone" in name.lower() or "cell phone" in name.lower() or "mobile" in name.lower():
                self.target_name = name
                break
        # 如果没有找到phone类，使用第一个类别
        if self.target_name is None and self.class_names:
            self.target_name = list(self.class_names.values())[0]
        print(f"告警目标类别: {self.target_name}")

        # 初始化本地数据库
        self._init_db()

    def _init_db(self):
        """初始化SQLite数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_base64 TEXT,
                detections TEXT,
                consecutive_frames INTEGER,
                alert_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        print(f"本地数据库已初始化: {self.db_path}")

    def _save_alert_local(self, alert_data: dict):
        """将告警数据保存到本地SQLite数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (timestamp, image_base64, detections, consecutive_frames, alert_type)
                VALUES (?, ?, ?, ?, ?)
            """, (
                alert_data["timestamp"],
                alert_data["image_base64"],
                json.dumps(alert_data["detections"], ensure_ascii=False),
                alert_data["consecutive_frames"],
                alert_data["alert_type"],
            ))
            conn.commit()
            conn.close()
            print(f"[本地存储] 告警已保存，ID: {cursor.lastrowid}")
        except Exception as e:
            print(f"[本地存储] 保存失败: {e}")

    def check_and_alert(self, detections: List[dict], frame: np.ndarray) -> bool:
        """
        检查检测结果，满足条件时发送告警
        判断逻辑: 检测到目标类别(手机)视为玩手机行为
        """
        has_target = any(d["class_name"] == self.target_name for d in detections) if self.target_name else False

        if has_target:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        if self.consecutive_count >= self.threshold:
            now = time.time()
            if now - self.last_alert_time > self.alert_cooldown:
                self._send_alert(detections, frame)
                self.last_alert_time = now
                self.consecutive_count = 0
                return True

        return False

    def _send_alert(self, detections: List[dict], frame: np.ndarray):
        """发送告警到服务器并保存到本地数据库"""
        try:
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_base64 = base64.b64encode(encoded).decode("utf-8")

            alert_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_base64": image_base64,
                "detections": detections,
                "consecutive_frames": self.consecutive_count,
                "alert_type": "person_playing_phone",
            }

            # 先保存到本地数据库
            self._save_alert_local(alert_data)

            # 再发送到远程服务器
            response = requests.post(
                f"{self.server_url}/api/alert",
                json=alert_data,
                timeout=10,
            )

            if response.status_code == 200:
                print(f"[告警] 已发送: {response.json()}")
            else:
                print(f"[告警] 发送失败: HTTP {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("[告警] 无法连接服务器，请检查网络")
        except Exception as e:
            print(f"[告警] 发送异常: {e}")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数 - 摄像头采集 + 本地推理 + 告警上传"""

    print("=" * 50)
    print("手机本地监控系统启动")
    print("=" * 50)

    # 初始化检测器
    print(f"加载模型: {MODEL_PATH}")
    detector = YOLODetector(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        imgsz=MODEL_IMGSZ,
    )

    # 初始化告警管理器
    alert_manager = AlertManager(
        threshold=CONSECUTIVE_FRAMES_THRESHOLD,
        server_url=SERVER_URL,
        class_names=detector.class_names,
    )

    # 打开摄像头
    print(f"打开摄像头 (索引={CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("摄像头已打开，开始检测...")
    print(f"告警服务器: {SERVER_URL}")
    print(f"连续帧阈值: {CONSECUTIVE_FRAMES_THRESHOLD}")
    print(f"跳帧检测: 每 {SKIP_FRAMES} 帧推理一次")
    print(f"模型输入尺寸: {MODEL_IMGSZ}")
    print("=" * 50)

    # 初始化tkinter窗口
    root = tk.Tk()
    root.title("手机本地监控")
    label = tk.Label(root)
    label.pack()

    frame_count = 0
    fps_start = time.time()
    last_detections = []  # 缓存上一帧的检测结果
    running = [True]  # 用列表保证闭包可修改
    fps = 0.0  # 初始化FPS

    def update_frame():
        if not running[0]:
            return

        ret, frame = cap.read()
        if not ret:
            running[0] = False
            root.destroy()
            return

        nonlocal frame_count, fps
        frame_count += 1

        # 跳帧检测
        if frame_count % SKIP_FRAMES == 0:
            detections = detector.detect(frame)
            last_detections.clear()
            last_detections.extend(detections)
        detections = last_detections.copy()

        # 绘制检测结果
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            lbl = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, lbl, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 连续帧计数
        cv2.putText(frame, f"Consecutive: {alert_manager.consecutive_count}/{CONSECUTIVE_FRAMES_THRESHOLD}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 告警判断
        alert_triggered = alert_manager.check_and_alert(detections, frame)
        if alert_triggered:
            cv2.putText(frame, "ALERT!", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # FPS统计 - 显示在画面上
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed
            fps_text = f"FPS: {fps:.1f} | Det: {len(detections)} | {', '.join(d['class_name'] for d in detections)}"
        else:
            fps_text = f"FPS: {fps:.1f} | Det: {len(detections)} | {', '.join(d['class_name'] for d in detections)}"

        cv2.putText(frame, fps_text, (10, frame.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 转换格式适配Tkinter
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_img)
        tk_img = ImageTk.PhotoImage(image=img)
        label.config(image=tk_img)
        label.image = tk_img

        root.after(30, update_frame)

    update_frame()
    root.mainloop()

    running[0] = False
    cap.release()
    print("监控系统已停止")


if __name__ == "__main__":
    main()
