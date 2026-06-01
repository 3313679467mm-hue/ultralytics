import os
import sys

# 添加项目根目录到 Python 路径，使用本地 ultralytics 代码
script_dir = os.path.dirname(__file__)
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

from ultralytics import YOLO

if __name__ == '__main__':
    # 获取脚本所在目录
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, '..', '..', 'datasets', 'phone_detection', 'data.yaml')
    
    # 使用原始 YOLOv8 配置
    model_yaml_path = os.path.join(script_dir, '..', '..', 'ultralytics', 'cfg', 'models', 'v8', 'yolov8.yaml')
    
    # 使用相同的预训练权重（保证公平对比）
    weights_path = os.path.join(script_dir, '..', '..', 'weights', 'best_modified.pt')
    
    print("=" * 60)
    print("GIoU 损失函数测试")
    print("=" * 60)
    print(f"配置文件：{model_yaml_path}")
    print(f"预训练权重：{weights_path}")
    print(f"数据集：{data_path}")
    print("=" * 60)
    
    # 创建模型并加载权重
    model = YOLO(model_yaml_path)
    model.load(weights_path)
    
    # 设置 IoU 类型参数
    model.args['iou_type'] = 'GIoU'
    
    # 开始训练
    print("\n开始训练 GIoU 模型...")
    results = model.train(
        # 数据集配置
        data=data_path,
        # 训练参数
        epochs=50,
        imgsz=640,
        device=0,  # GPU
        batch=4,
        # 数据增强参数（保持一致）
        hsv_h=0.025,
        hsv_s=0.8,
        hsv_v=0.5,
        fliplr=0.7,
        copy_paste=0.5,
        # 优化器参数
        patience=100,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # 损失函数权重
        box=7.5,
        cls=0.3,
        dfl=1.5,
        # IoU 类型 - GIoU
        iou_type='GIoU',
        # 其他参数
        dropout=0.0,
        name='phone_detection_giou',
        project='examples/phone_detection/runs',
        resume=False
    )
    print('训练完成!')
