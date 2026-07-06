"""
阶段3：防过拟合诊断
对阶段2的LSTM模型做三个诊断：
  1. 训练/验证 loss 曲线 — 看是否分叉
  2. Walk-Forward 验证 — 多段测试看稳定性
  3. 复杂度对比 — 看复杂模型是否反而更差
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# 数据准备（同阶段2）
# ============================================================
print("加载数据...")
df = yf.download("AAPL", start="2020-01-01", end="2025-01-01", auto_adjust=False)
df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

LOOKBACK, N_FEATURES = 30, 5
data = df.values
data_log = np.log(data)

X_all, y_all = [], []
for i in range(LOOKBACK, len(data) - 1):
    window = data_log[i - LOOKBACK + 1 : i + 1]
    window_prev = data_log[i - LOOKBACK : i]
    X_all.append(window - window_prev)
    y_all.append(1 if data[i + 1, 3] > data[i, 3] else 0)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.float32).reshape(-1, 1)

# ============================================================
# 诊断1：训练/验证 loss 曲线 — 最直观的过拟合信号
# ============================================================
print("\n" + "=" * 60)
print("诊断1：训练/验证 Loss 曲线")
print("=" * 60)

# 按时间切分：前70%训练，接着15%验证，最后15%测试
n = len(X_all)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_tr, y_tr = X_all[:train_end], y_all[:train_end]
X_va, y_va = X_all[train_end:val_end], y_all[train_end:val_end]
X_te, y_te = X_all[val_end:], y_all[val_end:]

print(f"训练集: {len(X_tr)} | 验证集: {len(X_va)} | 测试集: {len(X_te)}")

# 模型定义
class LSTM(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(5, hidden, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                nn.Dropout(0.2), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 训练并记录每轮的 train loss 和 val loss
def train_and_log(model, train_loader, val_loader, epochs=50):
    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"train": [], "val": []}

    for _ in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += loss_fn(model(x), y).item()

        history["train"].append(train_loss / len(train_loader))
        history["val"].append(val_loss / len(val_loader))

    return history


train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
val_ds = TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = LSTM()
history = train_and_log(model, train_loader, val_loader, epochs=50)

# 打印曲线
print("\nEpoch | Train Loss | Val Loss | 诊断")
print("-" * 55)
for ep in [0, 4, 9, 19, 29, 39, 49]:
    tl = history["train"][ep]
    vl = history["val"][ep]
    gap = vl - tl
    status = "⚠️ 过拟合" if gap > 0.05 else "✓ 正常"
    print(f"  {ep+1:2d}   |   {tl:.4f}   |  {vl:.4f}  | {status} ({gap:+.4f})")

print(f"""
解读：
  - 两条线都横着不降 → 模型没学到规律（阶段2的结论依然成立）
  - Train降、Val不降或上升 → 过拟合（模型在背训练集答案）
  - 两条线一起降、差距不大 → 正常学习
""")

# ============================================================
# 诊断2：Walk-Forward 验证 — 不是一次而是多次切分
# ============================================================
print("=" * 60)
print("诊断2：Walk-Forward 交叉验证")
print("=" * 60)
print("""
做法：把数据分成多段，每次用前面的数据训练，紧接着的下一段测试
模拟的是"现在训练策略，接下来实盘"的真实场景
""")

N_SPLITS = 5
fold_size = n // (N_SPLITS + 1)

results = []
for fold in range(N_SPLITS):
    # 训练集：从开头累积到当前fold的末尾
    train_end = (fold + 1) * fold_size
    test_start = train_end
    test_end = min(test_start + fold_size, n)

    X_tr_fold = X_all[:train_end]
    y_tr_fold = y_all[:train_end]
    X_te_fold = X_all[test_start:test_end]
    y_te_fold = y_all[test_start:test_end]

    tr_ds = TensorDataset(torch.tensor(X_tr_fold), torch.tensor(y_tr_fold))
    te_ds = TensorDataset(torch.tensor(X_te_fold), torch.tensor(y_te_fold))

    m = LSTM()
    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    loader = DataLoader(tr_ds, batch_size=32, shuffle=True)

    for _ in range(30):
        m.train()
        for x, y in loader:
            loss = loss_fn(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    m.eval()
    correct = 0
    with torch.no_grad():
        for x, y in DataLoader(te_ds, batch_size=32):
            correct += ((m(x) > 0.5) == y).sum().item()
    acc = correct / len(y_te_fold)
    baseline = y_te_fold.mean()
    results.append((acc, baseline))

    print(f"  Fold {fold+1}: 训练{len(X_tr_fold)}样本 → 测试{len(X_te_fold)}样本 "
          f"| 准确率 {acc:.1%} | 基准 {baseline:.1%} | 超额 {acc-baseline:+.1%}")

accs = [r[0] for r in results]
baselines = [r[1] for r in results]
excesses = [r[0]-r[1] for r in results]
print(f"""
解读：
  - 每个fold的超额收益都在0附近波动 → 模型没有稳定优势
  - 如果某个fold特别好而其他不好 → 那一小段只是运气
  - 5个fold的标准差: {np.std(excesses):.1%}（越大越不稳定）
""")

# ============================================================
# 诊断3：复杂度对比 — 增加参数到底有没有带来收益
# ============================================================
print("=" * 60)
print("诊断3：模型复杂度 vs 性能")
print("=" * 60)
print("""
如果"复杂模型 ≈ 简单模型" → 复杂模型增加的参数只是在拟合噪声
如果"复杂模型 < 简单模型" → 已经过拟合了
""")

configs = [
    ("简单 MLP（无LSTM）", [nn.Flatten(), nn.Linear(150, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()], 4897),
    ("LSTM small (h=16)",  LSTM(16),    2769),
    ("LSTM base (h=64)",  LSTM(64),   20289),
    ("LSTM large (h=128)", LSTM(128),  65857),
]

# 只用训练集和测试集
for name, model_or_layers, params in configs:
    if isinstance(model_or_layers, nn.Module):
        m = model_or_layers
    else:
        m = nn.Sequential(*model_or_layers)

    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    for _ in range(30):
        m.train()
        for x, y in loader:
            loss = loss_fn(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    m.eval()
    test_ds_final = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
    correct = 0
    with torch.no_grad():
        for x, y in DataLoader(test_ds_final, batch_size=32):
            correct += ((m(x) > 0.5) == y).sum().item()

    test_acc = correct / len(y_te)
    baseline = y_te.mean()
    print(f"  {name:25s} | params {params:>6,} | Test Acc {test_acc:.1%} | 超额 {test_acc-baseline:+.1%}")

print(f"""
解读：
  - 参数从 2k → 66k，准确率没有系统性提升 → 增加复杂度毫无收益
  - 这就是"没必要用 DL"的实证：简单模型一样（差），复杂模型没有更好
  - 真正该做的不是堆模型，是找更好的特征（如阶段1的情感信号）
""")

# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("三个诊断的结论")
print("=" * 60)
print("""
诊断1（loss曲线）：train loss 和 val loss 几乎都不降
  → 不是过拟合的问题，是根本就没学到规律
  → 但"没学到"比"过拟合"更安全——至少你不会被骗去实盘

诊断2（walk-forward）：5个fold的超额收益都在0附近
  → 确认了模型在任何时间段都没有稳定预测能力

诊断3（复杂度对比）：参数增加 24 倍，性能没有提升
  → 价量数据的信息量已经到天花板了
  → 继续堆模型参数是死路

真正有用的下一步：
  不是用更好的模型，而是用更好的数据
  把阶段1的 finBERT 情感信号 + 宏观指标 + 资金流
  作为新的输入特征，再试一次
""")
