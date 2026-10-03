from typing import Iterable

import pandas as pd

from ..engine.pipeline import Signal
from .index import INDEX_TICKERS
from .weights import adjust_weights

LOOKBACK_DAYS = 30      # ignore signals older than this
HALF_LIFE_DAYS = 7.0    # a signal's influence halves every 7 days


def _signal_frame(signals: Iterable[Signal]) -> pd.DataFrame:
    rows = []
    for s in signals:
        ts = pd.Timestamp(s.timestamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        rows.append({"ts": ts, "ticker": s.ticker,
                     "sentiment": s.sentiment, "impact": s.impact})
    return pd.DataFrame(rows, columns=["ts", "ticker", "sentiment", "impact"])


def decayed_sentiment(frame: pd.DataFrame, as_of: pd.Timestamp,
                      lookback: float = LOOKBACK_DAYS,
                      half_life: float = HALF_LIFE_DAYS) -> dict[str, float]:
    """Impact- and recency-weighted mean sentiment per ticker as of a moment."""
    out = {t: 0.0 for t in INDEX_TICKERS}
    if frame.empty:
        return out
    age = (as_of - frame["ts"]).dt.total_seconds() / 86400
    mask = (age >= 0) & (age <= lookback) & frame["ticker"].isin(INDEX_TICKERS)
    sub = frame[mask]
    if sub.empty:
        return out
    w = sub["impact"] * 0.5 ** (age[mask] / half_life)
    for ticker, g in sub.assign(w=w).groupby("ticker"):
        out[ticker] = float((g["sentiment"] * g["w"]).sum() / g["w"].sum())
    return out


def weight_history(signals: Iterable[Signal], dates) -> pd.DataFrame:
    """Daily index weights (rows = dates, columns = tickers) from replayed signals."""
    frame = _signal_frame(signals)
    rows = {}
    for d in dates:
        day = pd.Timestamp(d)
        as_of = day.tz_localize("UTC") + pd.Timedelta(days=1)   # end of that day
        rows[day] = adjust_weights(decayed_sentiment(frame, as_of))
    df = pd.DataFrame.from_dict(rows, orient="index")[INDEX_TICKERS]
    df.index.name = "date"
    return df
