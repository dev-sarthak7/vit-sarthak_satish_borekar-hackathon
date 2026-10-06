from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cross-encoder/nli-distilroberta-base"

# Event label -> natural-language description used in the NLI hypothesis
LABEL_DESCRIPTIONS = {
    "Credit Event": "a credit event such as a default, downgrade, bankruptcy or loan losses",
    "Geopolitical": "a geopolitical event such as war, sanctions, military conflict or political tension",
    "Macroeconomic": "a macroeconomic development such as inflation, interest rates, central bank policy or recession",
    "Merger/Acquisition": "a merger, acquisition or takeover",
    "Regulatory/Legal": "a regulatory or legal matter such as an investigation, lawsuit or fine",
    "Earnings": "company earnings, revenue or financial guidance",
    "Product Launch": "a new product launch or announcement",
    "Market Move": "a stock price move or market sell-off, without a specific underlying event",
    "Other": "routine company news with no major financial event",
}


@lru_cache(maxsize=1)
def _load():
    """Load the NLI model once and find which logits mean entailment/contradiction."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    names = {int(i): n.lower() for i, n in model.config.id2label.items()}
    # Fall back to this model's documented order: [contradiction, entailment, neutral]
    ent = next((i for i, n in names.items() if "entail" in n), 1)
    con = next((i for i, n in names.items() if "contra" in n), 0)
    return tokenizer, model, ent, con


def classify_event_zeroshot(text: str) -> tuple[str, float]:
    """Return (event_label, confidence in 0-1) for one text."""
    tokenizer, model, ent, con = _load()
    labels = list(LABEL_DESCRIPTIONS)
    hypotheses = [f"This news is about {LABEL_DESCRIPTIONS[l]}." for l in labels]

    enc = tokenizer([text] * len(hypotheses), hypotheses, padding=True,
                    truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits

    # Per label: P(entailment) against contradiction only
    probs = torch.softmax(logits[:, [con, ent]], dim=-1)[:, 1]
    best = int(torch.argmax(probs))
    return labels[best], float(probs[best])
