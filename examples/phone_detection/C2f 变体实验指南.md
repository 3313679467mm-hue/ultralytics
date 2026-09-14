# C2f 变体对比实验指南

## 实验目的

对比不同 C2f 变体模块在手机检测任务上的性能，验证注意力机制的有效性。

## 实验模块

### 1. C2f (原始版本)

- **配置文件**: `ultralytics/cfg/models/v8/yolov8.yaml`
- **训练脚本**: 已有的 `examples/phone_detection/train_simsppf.py` (使用原始 C2f)
- **特点**: 基础的 CSP 瓶颈结构，无注意力机制

### 2. C2fSE (融入 SE 注意力机制)

- **配置文件**: `ultralytics/cfg/models/v8/yolov8-c2fse.yaml`
- **训练脚本**: `examples/phone_detection/train_c2fse.py`
- **特点**: 在 C2f 的 Bottleneck 输出端添加 SE 注意力，自适应校准通道特征

### 3. C2fPSA (融入 PSA 注意力机制)

- **配置文件**: `ultralytics/cfg/models/v8/yolov8-c2fpsa.yaml`
- **训练脚本**: `examples/phone_detection/train_c2fpsa.py`
- **特点**: 使用 PSA 注意力机制，增强空间特征选择能力

## 实验步骤

### 步骤 1: 运行训练

依次运行以下命令进行训练：

```powershell
# 1. C2f (原始版本) - 已完成
D:\anaconda\envs\yolov8_new\python.exe examples\phone_detection\train_simsppf.py

# 2. C2fSE (SE 注意力)
D:\anaconda\envs\yolov8_new\python.exe examples\phone_detection\train_c2fse.py

# 3. C2fPSA (PSA 注意力)
D:\anaconda\envs\yolov8_new\python.exe examples\phone_detection\train_c2fpsa.py
```

### 步骤 2: 查看结果

训练完成后，运行对比脚本：

```powershell
D:\anaconda\envs\yolov8_new\python.exe examples\phone_detection\compare_c2f_variants.py
```

### 步骤 3: 填写实验报告

将对比结果填入下表：

| 模块                | P (%)    | R (%)    | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
| ------------------- | -------- | -------- | ----------- | ---------------- |
| C2f (原始版本)      | [待填写] | [待填写] | [待填写]    | [待填写]         |
| C2fSE (SE 注意力)   | [待填写] | [待填写] | [待填写]    | [待填写]         |
| C2fPSA (PSA 注意力) | [待填写] | [待填写] | [待填写]    | [待填写]         |
| 自主添加            | [待填写] | [待填写] | [待填写]    | [待填写]         |

## 进阶实验：探索不同替换位置

由于 C2f 模块在主干网络中多次使用，你可以尝试以下实验：

### 方案 1: 仅替换浅层 C2f

修改配置文件，只替换 P3/8 层的 C2f：

```yaml
backbone:
  - [-1, 3, C2fSE, [128, True]] # 仅第 2 层使用 C2fSE
  - [-1, 6, C2f, [256, True]] # 其他层使用原始 C2f
  - [-1, 6, C2f, [512, True]]
  - [-1, 3, C2f, [1024, True]]
```

### 方案 2: 仅替换深层 C2f

```yaml
backbone:
  - [-1, 3, C2f, [128, True]] # 浅层使用原始 C2f
  - [-1, 6, C2f, [256, True]]
  - [-1, 6, C2fSE, [512, True]] # P4/16 使用 C2fSE
  - [-1, 3, C2fSE, [1024, True]] # P5/32 使用 C2fSE
```

### 方案 3: 仅替换中间层

```yaml
backbone:
  - [-1, 3, C2f, [128, True]]
  - [-1, 6, C2fSE, [256, True]] # 仅 P3/8 使用 C2fSE
  - [-1, 6, C2f, [512, True]]
  - [-1, 3, C2f, [1024, True]]
```

### 方案 4: 混合使用不同注意力

```yaml
backbone:
  - [-1, 3, C2fSE, [128, True]] # 浅层用 SE
  - [-1, 6, C2fPSA, [256, True]] # 中层用 PSA
  - [-1, 6, C2fSE, [512, True]] # 深层用 SE
  - [-1, 3, C2fPSA, [1024, True]] # 最深用 PSA
```

## 实验结论分析要点

1. **注意力机制的有效性**：对比 C2f、C2fSE、C2fPSA 的性能差异
2. **通道注意力 vs 空间注意力**：SE 侧重通道，PSA 侧重空间
3. **计算复杂度**：注意力机制带来的参数量和计算量变化
4. **不同位置的影响**：浅层 vs 深层替换的效果对比

## 预期结果

- **C2fSE**：可能在 Precision 上有提升，SE 注意力能增强通道特征选择
- **C2fPSA**：可能在 Recall 上有提升，PSA 注意力能更好地定位目标
- **原始 C2f**：作为基准，性能应该最稳定

## 注意事项

1. 所有实验必须使用相同的：
   - 训练轮次 (50 epochs)
   - 批次大小 (batch=4)
   - 数据增强参数
   - 优化器参数
   - 预训练权重

2. 确保公平对比，每次训练前清空之前的结果

3. 记录训练时间和 GPU 利用率，评估计算效率
