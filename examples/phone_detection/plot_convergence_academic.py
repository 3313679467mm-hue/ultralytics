"""生成训练收敛曲线图 - 学术风格."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练结果
results = pd.read_csv('examples/phone_detection/runs/detect/train2/results.csv')

# 创建图表 - 白色背景学术风格
fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
ax.set_facecolor('white')

# 设置坐标轴颜色
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors='black')
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')

# 绘制 loss 曲线 (归一化到 0-0.4 范围)
box_loss = results['train/box_loss']
cls_loss = results['train/cls_loss']
dfl_loss = results['train/dfl_loss']
total_loss = box_loss + cls_loss + dfl_loss
loss_normalized = (total_loss - total_loss.min()) / (total_loss.max() - total_loss.min()) * 0.35 + 0.05

ax.plot(results['epoch'], loss_normalized, '--', color='blue', linewidth=1.5, alpha=0.8)
ax.text(10, 0.3, 'loss 曲线', color='blue', fontsize=12)

# 绘制 mAP@0.5 曲线
mAP50 = results['metrics/mAP50(B)']
ax.plot(results['epoch'], mAP50, '--', color='blue', linewidth=1.5, alpha=0.8)
ax.text(100, 0.95, 'mAP@0.5 精度曲线', color='blue', fontsize=12)

# 设置坐标轴
ax.set_xlim(0, 150)
ax.set_ylim(0, 1.0)
ax.set_xticks([0, 30, 60, 90, 120, 150])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# 添加坐标轴标签
ax.set_xlabel('迭代轮次 (Epochs)', color='black', fontsize=12)
ax.set_ylabel('指标数值', color='black', fontsize=12)

# 添加箭头
ax.annotate('', xy=(155, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.annotate('', xy=(0, 1.05), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()
plt.savefig('examples/phone_detection/训练收敛曲线_学术风格.png', dpi=300, bbox_inches='tight', facecolor='white')
print('学术风格图表已保存至: examples/phone_detection/训练收敛曲线_学术风格.png')
