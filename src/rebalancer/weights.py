from typing import Iterable, Optional

from ..engine.pipeline import Signal
from .index import INDEX_TICKERS, equal_weights

# How strongly sentiment moves a weight: multiplier = 1 + TILT * sentiment
TILT = 0.5
# Position limits (fraction of the index): no stock above 10% or below 3%
MIN_WEIGHT = 0.03
MAX_WEIGHT = 0.10


def ticker_sentiment(signals: Iterable[Signal]) -> dict[str, float]:
    """Impact-weighted mean sentiment per ticker (0.0 if a ticker has no signals)."""
    num = {t: 0.0 for t in INDEX_TICKERS}
    den = {t: 0.0 for t in INDEX_TICKERS}
    for s in signals:
        if s.ticker in num:
            num[s.ticker] += s.sentiment * s.impact
            den[s.ticker] += s.impact
    return {t: (num[t] / den[t] if den[t] else 0.0) for t in INDEX_TICKERS}


def apply_caps(
    weights: dict[str, float],
    min_w: float = MIN_WEIGHT,
    max_w: float = MAX_WEIGHT,
) -> dict[str, float]:
    """Clip weights to [min_w, max_w] and redistribute so they sum to exactly 1."""
    n = len(weights)
    if not (n * min_w <= 1 <= n * max_w):
        raise ValueError(f"caps infeasible for {n} stocks: min={min_w}, max={max_w}")

    w = {t: min(max(v, min_w), max_w) for t, v in weights.items()}
    for _ in range(100):
        diff = 1 - sum(w.values())
        if abs(diff) < 1e-12:
            break
        # Surplus goes to stocks still below the cap; a shortfall is taken
        # from stocks still above the floor.
        movable = [t for t, v in w.items() if (v < max_w if diff > 0 else v > min_w)]
        total = sum(w[t] for t in movable)
        for t in movable:
            w[t] = min(max(w[t] + diff * w[t] / total, min_w), max_w)
    return w


def adjust_weights(
    sentiment: dict[str, float],
    base: Optional[dict[str, float]] = None,
    tilt: float = TILT,
    min_w: float = MIN_WEIGHT,
    max_w: float = MAX_WEIGHT,
) -> dict[str, float]:
    """Tilt base weights toward positive sentiment, normalize, then apply caps."""
    base = base or equal_weights()
    raw = {t: base[t] * (1 + tilt * sentiment.get(t, 0.0)) for t in base}
    total = sum(raw.values())
    normalized = {t: w / total for t, w in raw.items()}
    return apply_caps(normalized, min_w, max_w)
