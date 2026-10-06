from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from ..ingestion.schema import Document
from .events import classify_event, event_scores
from .events_zeroshot import classify_event_zeroshot
from .impact import impact_score
from .sentiment import score_texts

# Zero-shot guesses below this need keyword support, otherwise they become "Other"
CONF_TRUST = 0.90
# Systemic labels (they can fire a stress test) must be backed by a keyword match
SYSTEMIC_EVENTS = ("Credit Event", "Geopolitical", "Macroeconomic")
# Confidence given to a label that came from the keyword fallback
KEYWORD_CONF = 0.6


class Signal(BaseModel):
    """Structured risk signal produced for one document."""
    timestamp: datetime
    ticker: Optional[str]
    source: str
    source_type: str
    text: str
    sentiment: float          # -1.0 to 1.0
    event: str                # event classification label
    event_confidence: float   # 0 to 1
    impact: float             # 1 to 10


def _resolve_event(text: str) -> tuple[str, float]:
    """Zero-shot label, cross-checked against the keyword baseline.

    - A systemic label needs at least one matching keyword; otherwise the
      keyword result is used instead.
    - A weak zero-shot guess with no keyword support at all becomes "Other".
    """
    label, conf = classify_event_zeroshot(text)
    keyword_label = classify_event(text)
    if label in SYSTEMIC_EVENTS and event_scores(text)[label] == 0:
        return keyword_label, KEYWORD_CONF
    if keyword_label == "Other" and conf < CONF_TRUST:
        return "Other", conf
    return label, conf


def process_documents(docs: list[Document]) -> list[Signal]:
    """Run sentiment, event classification and impact scoring on documents."""
    if not docs:
        return []
    sentiments = score_texts([d.text for d in docs])      # batched for speed

    signals = []
    for doc, sent in zip(docs, sentiments):
        event, conf = _resolve_event(doc.text)
        signals.append(Signal(
            timestamp=doc.timestamp,
            ticker=doc.ticker,
            source=doc.source,
            source_type=doc.source_type,
            text=doc.text,
            sentiment=round(sent, 4),
            event=event,
            event_confidence=round(conf, 4),
            impact=impact_score(sent, event, conf, doc.source_type),
        ))
    return signals
