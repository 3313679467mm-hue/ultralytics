"""
电脑本地监控 - YOLOv8实时目标检测.

功能:
- 调用电脑摄像头进行实时视频采集
- 使用冠军模型在本地进行推理
- 连续3帧检测到人玩手机行为时触发告警
- 将抓拍帧转为Base64编码，通过局域网POST发送至FastAPI后端

依赖安装:
    pip install opencv-python numpy requests ultralytics pillow
"""

from __future__ import annotations

import base64
import time
import tkinter as tk

import cv2
import numpy as np
import requests
from PIL import Image, ImageTk

from ultralytics import YOLO

# ============================================================
# 配置参数
# ============================================================

# 冠军模型路径
MODEL_PATH = "examples/phone_detection/runs/detect/train2/weights/best.pt"

# 告警后端API地址 (本机)
SERVER_URL = "http://127.0.0.1:8001"
ALERT_ENDPOINT = f"{SERVER_URL}/api/alert"

# 检测阈值
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# 连续帧触发告警的阈值
CONSECUTIVE_FRAMES_THRESHOLD = 3

# 摄像头索引 (0=默认摄像头)
CAMERA_INDEX = 0

# 跳帧检测 (每N帧推理一次, 提升FPS)
SKIP_FRAMES = 2

# 模型输入尺寸
MODEL_IMGSZ = 640


# ============================================================
# YOLODetector - YOLO推理引擎
# ============================================================


class YOLODetector:
    """YOLOv8推理引擎."""

    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45, imgsz: int = 640):
        """初始化检测器.

        Args:
            model_path: 模型文件路径 (.pt)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IOU阈值
            imgsz: 模型输入尺寸
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

    def detect(self, frame: np.ndarray) -> list[dict]:
        """对单帧图像进行目标检测.

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

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                        "confidence": confidence,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }
                )

        return detections


# ============================================================
# AlertManager - 告警管理器
# ============================================================


class AlertManager:
    """告警管理器 - 连续帧检测与告警上传."""

    def __init__(self, threshold: int = 3, server_url: str = "http://127.0.0.1:8001", class_names: dict | None = None):
        self.threshold = threshold
        self.server_url = server_url
        self.consecutive_count = 0
        self.last_alert_time = 0
        self.alert_cooldown = 10  # 告警冷却时间(秒)
        self.class_names = class_names or {}
        # 查找目标类别名称 (手机)
        self.target_name = None
        for cid, name in self.class_names.items():
            if "phone" in name.lower() or "cell phone" in name.lower() or "mobile" in name.lower():
                self.target_name = name
                break
        # 如果没有找到phone类，使用第一个类别
        if self.target_name is None and self.class_names:
            self.target_name = next(iter(self.class_names.values()))
        print(f"告警目标类别: {self.target_name}")

    def check_and_alert(self, detections: list[dict], frame: np.ndarray) -> bool:
        """检查检测结果，满足条件时发送告警 判断逻辑: 检测到目标类别(手机)视为玩手机行为.
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

    def _send_alert(self, detections: list[dict], frame: np.ndarray):
        """发送告警到服务器."""
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
    """主函数 - 摄像头采集 + 本地推理 + 告警上传."""
    print("=" * 50)
    print("电脑本地监控系统启动")
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
    root.title("电脑本地监控")
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
            cv2.putText(frame, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 连续帧计数
        cv2.putText(
            frame,
            f"Consecutive: {alert_manager.consecutive_count}/{CONSECUTIVE_FRAMES_THRESHOLD}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # 告警判断
        alert_triggered = alert_manager.check_and_alert(detections, frame)
        if alert_triggered:
            cv2.putText(frame, "ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # FPS统计 - 显示在画面上
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed
        fps_text = f"FPS: {fps:.1f} | Det: {len(detections)} | {', '.join(d['class_name'] for d in detections)}"

        cv2.putText(frame, fps_text, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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
