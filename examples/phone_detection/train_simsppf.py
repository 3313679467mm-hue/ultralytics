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
    
    # SimSPPF 配置文件路径
    model_yaml_path = os.path.join(script_dir, '..', '..', 'ultralytics', 'cfg', 'models', 'v8', 'yolov8-simsppf.yaml')
    
    # 使用相同的预训练权重（保证公平对比）
    weights_path = os.path.join(script_dir, '..', '..', 'weights', 'best_modified.pt')
    
    print("=" * 60)
    print("SimSPPF 模块测试")
    print("=" * 60)
    print(f"配置文件：{model_yaml_path}")
    print(f"预训练权重：{weights_path}")
    print(f"数据集：{data_path}")
    print("=" * 60)
    
    # 创建模型并加载权重
    model = YOLO(model_yaml_path)
    model.load(weights_path)
    
    # 开始训练
    print("\n开始训练 SimSPPF 模型...")
    results = model.train(
        # 数据集配置
        data=data_path,
        # 训练参数（与 SPPF 测试完全一致）
        epochs=50,
        imgsz=640,
        device=0,
        batch=4,
        # 结果保存
        name='phone_detection_simsppf',
        project='runs',
        # 数据增强
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
        dropout=0.0,
        resume=False,
        # 低内存配置
        workers=0,
        amp=False,
        cache=False
    )
    
    print('\n' + '=' * 60)
    print('SimSPPF 训练完成!')
    print('=' * 60)
    print(f'结果保存在：runs/detect/phone_detection_simsppf/')
    print('\n对比说明:')
    print('  - train.py: 使用默认 SPPF')
    print('  - train_spp.py: 使用自定义 SPP 配置')
    print('  - train_simsppf.py: 使用 SimSPPF 配置')
    print('  - 所有训练参数完全一致，确保公平对比')
    print('=' * 60)
