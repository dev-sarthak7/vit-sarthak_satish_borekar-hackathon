from typing import Iterable, Optional

from ..engine.pipeline import Signal
from .index import INDEX_TICKERS, equal_weights

# How strongly sentiment moves a weight: multiplier = 1 + TILT * sentiment
TILT = 0.5


def ticker_sentiment(signals: Iterable[Signal]) -> dict[str, float]:
    """Impact-weighted mean sentiment per ticker (0.0 if a ticker has no signals)."""
    num = {t: 0.0 for t in INDEX_TICKERS}
    den = {t: 0.0 for t in INDEX_TICKERS}
    for s in signals:
        if s.ticker in num:
            num[s.ticker] += s.sentiment * s.impact
            den[s.ticker] += s.impact
    return {t: (num[t] / den[t] if den[t] else 0.0) for t in INDEX_TICKERS}


def adjust_weights(
    sentiment: dict[str, float],
    base: Optional[dict[str, float]] = None,
    tilt: float = TILT,
) -> dict[str, float]:
    """Tilt base weights toward positive sentiment, then renormalize to sum to 1."""
    base = base or equal_weights()
    raw = {t: base[t] * (1 + tilt * sentiment.get(t, 0.0)) for t in base}
    total = sum(raw.values())
    return {t: w / total for t, w in raw.items()}
