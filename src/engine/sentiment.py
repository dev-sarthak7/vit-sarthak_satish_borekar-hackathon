from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def _load():
    """Load the model once and reuse it for every call."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def score_texts(texts: list[str], batch_size: int = 16) -> list[float]:
    """Return a sentiment score in [-1.0, 1.0] for each text.

    score = P(positive) - P(negative), using FinBERT's class probabilities.
    """
    tokenizer, model = _load()
    labels = {i: name.lower() for i, name in model.config.id2label.items()}
    pos = next(i for i, name in labels.items() if name == "positive")
    neg = next(i for i, name in labels.items() if name == "negative")

    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=256, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        scores.extend((probs[:, pos] - probs[:, neg]).tolist())
    return scores


def score_text(text: str) -> float:
    return score_texts([text])[0]
