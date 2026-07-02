"""
阶段1：NLP 情感分析 — finBERT 金融新闻情感打分
层次A：不写模型，直接用预训练模型推理
"""
from transformers import pipeline

# finBERT：专门在金融文本上微调过的 BERT
# 首次运行会自动下载模型（约 500MB），之后会用缓存
classifier = pipeline(
    "text-classification",
    model="ProsusAI/finBERT",
    return_all_scores=True,
)

# 测试：几条金融新闻标题
news = [
    "Fed signals rate cuts likely in September, markets rally",
    "US debt ceiling crisis deepens, default risk looms",
    "Apple reports record quarterly revenue, beats expectations",
    "Oil prices crash 15% on demand fears and oversupply",
    "Market closes flat in mixed trading session",
]

print("=== 金融新闻情感打分 ===\n")
for headline in news:
    scores = classifier(headline)[0]
    # scores 格式：[{label: 'positive', score: 0.7}, {label: 'negative', score: 0.05}, {label: 'neutral', score: 0.25}]
    pos = next(s["score"] for s in scores if s["label"] == "positive")
    neg = next(s["score"] for s in scores if s["label"] == "negative")
    neu = next(s["score"] for s in scores if s["label"] == "neutral")

    # 合成一个分数：正=+1, 负=-1, 中性=0，用概率加权
    signal = pos * 1.0 + neg * (-1.0) + neu * 0.0

    print(f"📰 {headline}")
    print(f"   正面 {pos:.2%}  负面 {neg:.2%}  中性 {neu:.2%}")
    print(f"   → 信号得分: {signal:+.3f}\n")
