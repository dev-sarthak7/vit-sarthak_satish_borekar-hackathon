from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import Document
from .tickers import UNIVERSE, tag_ticker

# Column names we accept (case-insensitive) for each field
TEXT_COLS = ["tweet", "text", "content"]
TIME_COLS = ["date", "timestamp", "created_at"]
TICKER_COLS = ["stock name", "ticker", "symbol", "stock"]

# Datasets sometimes use a different share class / symbol than our universe
ALIASES = {"GOOGL": "GOOG"}


def _find_col(columns, candidates) -> Optional[str]:
    lowered = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def load_tweets_csv(path, limit: Optional[int] = None) -> list[Document]:
    """Load tweets from a CSV and normalize them to Document.

    Keeps only tweets that map to a ticker in our index universe.
    """
    df = pd.read_csv(Path(path))
    text_col = _find_col(df.columns, TEXT_COLS)
    time_col = _find_col(df.columns, TIME_COLS)
    ticker_col = _find_col(df.columns, TICKER_COLS)

    if text_col is None or time_col is None:
        raise ValueError(
            f"Could not find text/date columns. Columns present: {list(df.columns)}"
        )

    df["_ts"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=["_ts", text_col])
    if limit:
        df = df.head(limit)

    docs = []
    for _, row in df.iterrows():
        text = str(row[text_col])
        if ticker_col is not None:
            # Trust the dataset's own label; never re-tag by company name
            raw = str(row[ticker_col]).strip().upper()
            ticker = ALIASES.get(raw, raw)
            if ticker not in UNIVERSE:
                continue                  # keep only tweets about our index
        else:
            ticker = tag_ticker(text)
            if ticker is None:
                continue
        try:
            docs.append(Document(
                source="kaggle_tweets", source_type="social",
                timestamp=row["_ts"].to_pydatetime(), text=text, ticker=ticker,
            ))
        except ValueError:
            continue                      # skip malformed rows
    return docs
