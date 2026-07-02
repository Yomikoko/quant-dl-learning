"""
阶段2：PyTorch 基础 — LSTM 价格涨跌预测
层次B：自己写模型、训练循环、看 loss 下降
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# ============================================================
# 1. 准备数据：用 yfinance 拉真实股票数据
# ============================================================
print("1. 加载数据...")
import yfinance as yf

# 拉取苹果股票 5 年日线数据
df = yf.download("AAPL", start="2020-01-01", end="2025-01-01", auto_adjust=False)
df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

print(f"   数据量: {len(df)} 天")
print(f"   日期范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

# ============================================================
# 2. 特征工程：用过去 30 天的 OHLCV 预测下一天涨跌
# ============================================================
print("\n2. 构造训练样本...")

LOOKBACK = 30          # 看过去 30 天
N_FEATURES = 5         # O, H, L, C, V

# 原始数据转成收益率（让不同时期的价格可比）
data = df.values  # shape: (n_days, 5)
data_log = np.log(data)

# 特征：过去30天每天的收益率 (相对于前一天)
features = []
labels = []

for i in range(LOOKBACK, len(data) - 1):
    # 特征：最近 30 天的 log收益率
    window = data_log[i - LOOKBACK + 1 : i + 1]          # 30 days
    window_prev = data_log[i - LOOKBACK : i]             # 30 days offset by 1
    returns = window - window_prev                        # day-to-day returns
    features.append(returns)

    # 标签：下一天涨(1)还是跌(0)
    next_close = data[i + 1, 3]   # Close price
    curr_close = data[i, 3]
    labels.append(1 if next_close > curr_close else 0)

X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.float32).reshape(-1, 1)

n_samples = len(X)
print(f"   样本数: {n_samples}")
print(f"   特征形状: {X.shape}  → (样本数, 时间步{LOOKBACK}, 特征数{N_FEATURES})")
print(f"   涨占比: {y.mean():.1%}")

# 按时间切分：前 80% 训练，后 20% 测试（绝对不能随机打乱！）
split = int(n_samples * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"   训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
print(f"   ⚠️ 按时间切分，先训练后测试，防止未来信息泄漏")

# 转成 PyTorch Tensor 和 DataLoader
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练集可以打乱
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)   # 测试集不能打乱

# ============================================================
# 3. 定义模型（五个组件全在这一个类里）
# ============================================================
print("\n3. 定义 LSTM 模型...")


class PriceDirectionLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super().__init__()
        # ② 特征提取器：LSTM 读时间序列
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,   # 输入格式 (batch, seq_len, features)
            dropout=0.2,        # dropout 只在 num_layers>=2 时生效
        )
        # ③ 输出头：把 hidden state 映射到涨跌概率
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),       # 压到 (0,1)，解释为涨的概率
        )

    def forward(self, x):
        # x 形状: (batch, 30, 5)
        out, (hidden, cell) = self.lstm(x)
        # out[:, -1] → 取最后一个时间步的输出
        return self.fc(out[:, -1, :])


model = PriceDirectionLSTM()
total_params = sum(p.numel() for p in model.parameters())
print(f"   模型参数量: {total_params:,}")

# ④ 损失函数：二分类用 BCE
loss_fn = nn.BCELoss()
# ⑤ 优化器：Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# 4. 训练循环
# ============================================================
print("\n4. 开始训练...\n")
EPOCHS = 30

for epoch in range(EPOCHS):
    # === 训练阶段 ===
    model.train()
    train_loss = 0.0
    train_correct = 0

    for x_batch, y_batch in train_loader:          # ① 取一批数据
        y_pred = model(x_batch)                     # ②③ 前向传播
        loss = loss_fn(y_pred, y_batch)             # ④ 算损失
        optimizer.zero_grad()
        loss.backward()                              # 反向传播
        optimizer.step()                             # ⑤ 更新参数

        train_loss += loss.item()
        train_correct += ((y_pred > 0.5) == y_batch).sum().item()

    train_acc = train_correct / len(y_train)

    # === 验证阶段（测试集，不更新参数）===
    model.eval()
    test_loss = 0.0
    test_correct = 0

    with torch.no_grad():  # 关掉梯度计算，省显存
        for x_batch, y_batch in test_loader:
            y_pred = model(x_batch)
            test_loss += loss_fn(y_pred, y_batch).item()
            test_correct += ((y_pred > 0.5) == y_batch).sum().item()

    test_acc = test_correct / len(y_test)

    print(
        f"  Epoch {epoch+1:2d}/{EPOCHS}  "
        f"Train Loss: {train_loss/len(train_loader):.4f}  "
        f"Train Acc: {train_acc:.1%}  "
        f"Test Acc: {test_acc:.1%}"
    )

# ============================================================
# 5. 结果分析
# ============================================================
print(f"\n5. 最终结果")
print(f"   测试集准确率: {test_acc:.1%}")
print(f"   基准（猜涨）: {y_test.mean().item():.1%}")
print(f"   超额: {test_acc - y_test.mean().item():+.1%}")
print(f"\n   ⚠️ 如果准确率很高（>60%），先别高兴——大概率有过拟合或未来信息泄漏。")
print(f"   下一个阶段会系统学习如何诊断和防范。")
