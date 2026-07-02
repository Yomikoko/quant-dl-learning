# 阶段2 笔记：PyTorch 基础 — LSTM 价格预测

## 学到了什么

### 完整训练流程

```python
# 1. 数据 → Tensor → DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 定义模型（继承 nn.Module）
class MyModel(nn.Module):
    def __init__(self):   # 定义层
    def forward(self, x): # 定义前向流程

# 3. 训练循环
for epoch in range(EPOCHS):
    for x, y in train_loader:
        pred = model(x)          # 前向
        loss = loss_fn(pred, y)  # 算损失
        loss.backward()          # 自动求梯度
        optimizer.step()         # 更新参数
```

### 金融数据的特殊处理

1. **用收益率而非原始价格**：价格从 $20→$150，模型没法跨时期泛化
2. **按时间切分而非随机切分**：否则测试集数据可能出现在训练集之前 → 未来信息泄漏
3. **测试集绝不能 shuffle**：保持真实的时间顺序

### 关键结果：价格预测非常难

- BCE Loss 的随机基线 ≈ 0.693（-ln(0.5)）
- 训练几乎没降低 loss → 模型没学到有效规律
- 测试准确率（54.9%）低于"永远猜涨"（57.3%）

### 为什么会这样

| 原因 | 说明 |
|------|------|
| OHLCV 信息量太低 | 价量是结果，不是原因。真正驱动价格的是信息流 |
| 短期价格近似随机游走 | EMH 有效市场假说：明天的涨跌 ≈ 抛硬币 |
| 30 天太短 | 不足以学到长期周期 |
| 缺特征 | 没有舆情、宏观、资金流等辅助信息 |

### 启示

DL 在量化里的正确姿势不是"裸价量预测涨跌"，而是：
- 结合 NLP 情感信号
- 预测波动率（比方向好预测）
- 做因子合成和组合优化
- 异常检测（识别市场状态切换）

## 新学的函数/概念

| 概念 | 代码 |
|------|------|
| 张量转换 | `torch.tensor(arr, dtype=torch.float32)` |
| 数据集封装 | `TensorDataset(X_tensor, y_tensor)` |
| 批次加载 | `DataLoader(dataset, batch_size=32, shuffle=True)` |
| 训练模式 | `model.train()` → 启用 dropout |
| 推理模式 | `model.eval()` → 关闭 dropout |
| 无梯度上下文 | `with torch.no_grad():` |
