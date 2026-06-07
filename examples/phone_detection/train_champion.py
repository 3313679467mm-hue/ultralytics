import os

from ultralytics import YOLO

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, "..", "..", "datasets", "phone_detection", "data.yaml")
    model_yaml_path = os.path.join(
        script_dir, "..", "..", "ultralytics", "cfg", "models", "v8", "yolov8-phone-final.yaml"
    )
    weights_path = os.path.join(script_dir, "..", "..", "weights", "yolov8n.pt")

    # 1. 载入终极冠军架构模型（自动下载 yolov8n.pt 预训练权重）
    model = YOLO(model_yaml_path)

    # 3. 启动终极对比训练 (显式传递已在 default.yaml 中修复注册的 iou_type)
    model.train(
        data=data_path,  # 数据集路径
        epochs=150,  # 收敛深度建议拉长到 100-150
        batch=4,  # 根据实验平台保持一致
        iou_type="SIoU",  # 强制指定引入了角度惩罚的 SIoU
        device=0,  # 指定 GPU
        workers=4,
    )
