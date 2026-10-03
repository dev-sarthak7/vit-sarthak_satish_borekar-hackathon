from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from .index import INDEX_TICKERS

PRICES_PATH = Path("data/prices.csv")
# Matches the date range of the stock tweets dataset
START, END = "2021-09-30", "2022-09-30"


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """One sequential Yahoo Finance download (threads=False avoids cache locks)."""
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices.reindex(columns=tickers)


def fetch_prices(tickers: Optional[list[str]] = None,
                 start: str = START, end: str = END) -> pd.DataFrame:
    """Download daily adjusted closes, retrying tickers that come back empty."""
    tickers = tickers or INDEX_TICKERS
    prices = _download(tickers, start, end)

    for _ in range(3):
        missing = [t for t in tickers if prices[t].isna().all()]
        if not missing:
            break
        print(f"[prices] retrying {missing}")
        prices[missing] = _download(missing, start, end)

    missing = [t for t in tickers if prices[t].isna().all()]
    if missing:
        raise RuntimeError(f"no price data for: {missing}")

    prices = prices.dropna(how="all")
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    return prices


def load_prices(refresh: bool = False, path: Path = PRICES_PATH) -> pd.DataFrame:
    """Use the cached CSV when present; otherwise download and cache."""
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="date", parse_dates=True)
    try:
        prices = fetch_prices()
    except Exception as e:
        if path.exists():
            print(f"[prices] download failed ({e}); using cached CSV")
            return pd.read_csv(path, index_col="date", parse_dates=True)
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    return prices


if __name__ == "__main__":
    df = load_prices(refresh=True)
    print(f"{df.shape[0]} days x {df.shape[1]} tickers, "
          f"{df.index.min().date()} to {df.index.max().date()}, "
          f"{int(df.isna().sum().sum())} missing values")
