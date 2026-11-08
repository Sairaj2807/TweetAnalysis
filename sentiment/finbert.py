"""Loads FinBERT once and scores batches of text into (positive, negative, neutral) probs."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
LABEL_ORDER = ["positive", "negative", "neutral"]

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def score_texts(texts: list[str], batch_size: int = 32, max_length: int = 64) -> list[tuple[float, float, float]]:
    """Returns a list of (positive, negative, neutral) probability tuples."""
    tokenizer, model = _load()
    results = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
            results.extend([(float(p[0]), float(p[1]), float(p[2])) for p in probs])
    return results
