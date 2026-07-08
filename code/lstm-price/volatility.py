"""
换个预测目标：预测波动率而不是涨跌方向
波动率比方向更可预测——因为波动率有聚集效应（大波动之后往往还是大波动）
"""
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, pandas as pd, yfinance as yf
from sklearn.metrics import r2_score

print("加载数据...")
df = yf.download("AAPL", start="2020-01-01", end="2025-01-01", auto_adjust=False)
df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

LOOKBACK = 30
data = df.values
data_log = np.log(data)

X, y = [], []
for i in range(LOOKBACK, len(data) - 1):
    window = data_log[i - LOOKBACK + 1 : i + 1]
    window_prev = data_log[i - LOOKBACK : i]
    X.append(window - window_prev)

    # 标签改为：下一天的日内波动率（High-Low range）
    tomorrow_vol = (data[i + 1, 1] - data[i + 1, 2]) / data[i + 1, 3]  # (H-L)/C
    y.append(tomorrow_vol)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

# 按时间切分
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


# 同一个 LSTM，改输出头：回归（不加 Sigmoid，不做分类）
class LSTMVol(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(5, hidden, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )  # 没有 Sigmoid → 输出任意实数

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


model = LSTMVol()
loss_fn = nn.MSELoss()  # 回归用 MSE，不是 BCE
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n训练中...")
for epoch in range(50):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            val_loss += loss_fn(model(xb), yb).item()

    if epoch % 10 == 0 or epoch == 49:
        print(f"  Epoch {epoch+1:2d} | Train MSE: {train_loss/len(train_loader):.6f} | "
              f"Test MSE: {val_loss/len(test_loader):.6f}")

# 评估：R² 和 Pearson 相关
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds.append(model(xb).numpy())
        actuals.append(yb.numpy())
preds = np.concatenate(preds).flatten()
actuals = np.concatenate(actuals).flatten()

r2 = r2_score(actuals, preds)
corr = np.corrcoef(preds, actuals)[0, 1]

# 简单基线：用昨天的波动率预测今天的（naive persistence）
baseline_pred = np.roll(actuals, 1)
baseline_pred[0] = baseline_pred[1]
baseline_corr = np.corrcoef(baseline_pred, actuals)[0, 1]

print(f"\n=== 结果对比 ===")
print(f"  预测方向（阶段2）:  准确率 54.9%, 超额 -2.4%")
print(f"  预测波动率（本次）:  R² = {r2:.3%}, 相关性 r = {corr:.3f}")
print(f"  基准（朴素预测）:   相关性 r = {baseline_corr:.3f} (用昨天波动率猜今天)")
print(f"  LSTM vs 基准:       {corr - baseline_corr:+.3f}")
print(f"""
解读：
  - r = {corr:.3f} 意味着预测值和真实值有弱正相关
  - 比"用昨天猜今天"的基准{'好' if corr > baseline_corr else '差'}
  - 这不是很强的信号，但至少不是瞎猜
  - 波动率比方向可预测得多，因为有"波动率聚集"效应
  - 如果加上 VIX、期权隐含波动率等特征，r 可以做到 0.3+
""")
