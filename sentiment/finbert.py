"""Loads FinBERT once and scores batches of text into (positive, negative, neutral) probs."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
LABEL_ORDER = ["positive", "negative", "neutral"]

_tokenizer = None
_model = None
_device = None


def _load():
    global _tokenizer, _model, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        _model.to(_device)
        print(f"  FinBERT loaded on device: {_device}", flush=True)
    return _tokenizer, _model


def score_texts(texts: list[str], batch_size: int = 64, max_length: int = 64, progress_every: int = 50) -> list[tuple[float, float, float]]:
    """Returns a list of (positive, negative, neutral) probability tuples."""
    tokenizer, model = _load()
    results = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
            results.extend([(float(p[0]), float(p[1]), float(p[2])) for p in probs])
            if progress_every and (batch_idx + 1) % progress_every == 0:
                print(f"    scored {i + len(batch)}/{len(texts)} texts ({batch_idx + 1}/{n_batches} batches)", flush=True)
    return results
