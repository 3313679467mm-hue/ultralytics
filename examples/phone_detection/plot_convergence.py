"""生成训练收敛曲线图."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 读取训练结果
results = pd.read_csv('examples/phone_detection/runs/detect/train2/results.csv')

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('冠军模型 (SimSPPF+SIoU+C2fCBAM+C2fLightConv) 150轮训练收敛曲线', fontsize=16, fontweight='bold')

# 1. 训练损失曲线
ax = axes[0, 0]
ax.plot(results['epoch'], results['train/box_loss'], label='Box Loss', linewidth=2)
ax.plot(results['epoch'], results['train/cls_loss'], label='Cls Loss', linewidth=2)
ax.plot(results['epoch'], results['train/dfl_loss'], label='DFL Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('训练损失曲线', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 验证损失曲线
ax = axes[0, 1]
ax.plot(results['epoch'], results['val/box_loss'], label='Box Loss', linewidth=2)
ax.plot(results['epoch'], results['val/cls_loss'], label='Cls Loss', linewidth=2)
ax.plot(results['epoch'], results['val/dfl_loss'], label='DFL Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('验证损失曲线', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 精度指标曲线
ax = axes[0, 2]
ax.plot(results['epoch'], results['metrics/precision(B)'], label='Precision', linewidth=2)
ax.plot(results['epoch'], results['metrics/recall(B)'], label='Recall', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('精度指标曲线', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. mAP曲线
ax = axes[1, 0]
ax.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='green')
ax.plot(results['epoch'], results['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='orange')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('mAP', fontsize=12)
ax.set_title('mAP收敛曲线', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 5. 学习率曲线
ax = axes[1, 1]
ax.plot(results['epoch'], results['lr/pg0'], label='PG0', linewidth=1.5)
ax.plot(results['epoch'], results['lr/pg1'], label='PG1', linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('学习率衰减曲线', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 综合指标
ax = axes[1, 2]
ax.plot(results['epoch'], results['metrics/precision(B)'], label='Precision', linewidth=2)
ax.plot(results['epoch'], results['metrics/recall(B)'], label='Recall', linewidth=2)
ax.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('综合性能指标', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/phone_detection/训练收敛曲线.png', dpi=300, bbox_inches='tight')
print('图表已保存至: examples/phone_detection/训练收敛曲线.png')
