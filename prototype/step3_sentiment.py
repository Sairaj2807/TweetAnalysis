import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL = "ProsusAI/finbert"

print("Loading FinBERT (first run downloads ~400MB)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

# FinBERT label order: 0=positive, 1=negative, 2=neutral
LABELS = ["positive", "negative", "neutral"]

tweets = pd.read_csv("tweets_data.csv", parse_dates=["time"])

sentiments = []
batch_size = 32
texts = tweets["text"].tolist()

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
        for p in probs:
            score = p[0] - p[1]  # positive - negative
            sentiments.append(score)
        print(f"Scored {min(i + batch_size, len(texts))}/{len(texts)}")

tweets["sentiment"] = sentiments
tweets.to_csv("tweets_scored.csv", index=False)
print("Saved tweets_scored.csv")
print(tweets.head())
