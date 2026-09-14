import os

from ultralytics import YOLO

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(__file__)
    weights_dir = os.path.join(script_dir, "..", "..", "weights")
    data_path = os.path.join(script_dir, "..", "..", "datasets", "phone_detection", "data.yaml")

    # 自定义 SPP 配置文件路径（唯一区别：使用 yolov8-spp.yaml 而非默认配置）
    model_yaml_path = os.path.join(script_dir, "..", "..", "ultralytics", "cfg", "models", "v8", "yolov8-spp.yaml")

    # 使用与 train.py 相同的预训练权重
    model_path = os.path.join(weights_dir, "best_modified.pt")

    # 1. 加载自定义配置文件并加载预训练权重
    print(f"加载自定义配置文件: {model_yaml_path}")
    print(f"加载预训练权重: {model_path}")
    print(f"数据集配置: {data_path}")

    # 创建模型并加载权重（使用自定义 SPP 配置）
    model = YOLO(model_yaml_path)  # 使用 yolov8-spp.yaml
    model.load(model_path)  # 加载相同权重

    # 2. 开始训练 - 使用与 train.py 完全相同的配置
    print("\n开始训练...")
    results = model.train(
        # 数据集配置
        data=data_path,  # 数据集配置文件路径
        # 训练参数（与 train.py 完全一致）
        epochs=50,  # 训练轮数 - 相同
        imgsz=640,  # 图像尺寸 - 相同
        device=0,  # 使用 GPU 0 - 相同
        batch=4,  # 批大小 - 相同
        # 结果保存（使用不同的 name 以便区分）
        name="phone_detection_spp",  # 实验名称 - 不同（用于区分）
        project="runs",  # 项目根目录 - 相同
        # 数据增强（与 train.py 完全一致）
        hsv_h=0.025,  # 色调增强强度 - 相同
        hsv_s=0.8,  # 饱和度增强强度 - 相同
        hsv_v=0.5,  # 明度增强强度 - 相同
        fliplr=0.7,  # 水平翻转概率 - 相同
        copy_paste=0.5,  # 复制粘贴增强概率 - 相同
        # 优化器参数（与 train.py 完全一致）
        patience=100,  # 早停耐心值 - 相同
        warmup_epochs=3.0,  # 预热轮数 - 相同
        warmup_momentum=0.8,  # 预热动量 - 相同
        warmup_bias_lr=0.1,  # 预热偏置学习率 - 相同
        # 损失函数权重（与 train.py 完全一致）
        box=7.5,  # 边界框损失权重 - 相同
        cls=0.3,  # 分类损失权重 - 相同
        dfl=1.5,  # 分布焦点损失权重 - 相同
        dropout=0.0,  # dropout 概率 - 相同
        resume=False,  # 不从之前的训练状态恢复 - 相同
    )

    print("\n训练完成!")
    print("结果保存在：runs/detect/phone_detection_spp/")
    print("\n对比说明：")
    print("  - train.py: 使用默认 YOLOv8n 配置")
    print("  - train_spp.py: 使用自定义 yolov8-spp.yaml 配置")
    print("  - 其他所有训练参数完全一致，确保公平对比")
