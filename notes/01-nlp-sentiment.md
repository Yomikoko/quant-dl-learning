# 阶段1 笔记：NLP 情感分析

## 学到了什么

### 核心概念

- **预训练模型**：别人在大规模数据上训好的模型，你直接下载用 → 不需要自己训练
- **finBERT**：在金融新闻/财报文本上微调过的 BERT，比通用 BERT 更懂金融语境
  - 普通 BERT 看到 "short" → 理解为"矮的/短的"
  - finBERT 看到 "short" → 理解为"做空"
- **推理 (Inference)**：模型已经训好了，你只做"输入 → 模型 → 输出"

### 工作流

```
新闻标题 → finBERT → 三分类概率(正面/负面/中性) → 加权合成信号得分
```

### 信号得分公式

```
signal = P(正面) × 1.0 + P(负面) × (-1.0) + P(中性) × 0.0
```

结果范围 [-1, 1]，正数看多，负数看空。可批量处理每日新闻，每日汇总得到情感因子。

### HuggingFace pipeline

```python
pipeline("text-classification", model="ProsusAI/finBERT")
```
pipeline() 是 HuggingFace 的一键调用接口，自动处理：
- 下载模型权重
- 加载 tokenizer（把文本切成 token）
- 推理
- 输出可读结果

不需要知道 Transformer 内部怎么算的（层次 A 到此为止）。

## 下一层

层次 B：用 PyTorch 搭自己的模型，理解 forward() 和训练循环。
