import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from .schema import Document
from .tickers import UNIVERSE, tag_ticker

load_dotenv()

NEWSAPI_URL = "https://newsapi.org/v2/everything"
DEFAULT_CSV = Path("data/sample_news.csv")


def fetch_newsapi(ticker: str, page_size: int = 20) -> list[Document]:
    """Fetch recent headlines for one ticker from NewsAPI."""
    key = os.getenv("NEWSAPI_KEY")
    if not key or key == "your_key_here":
        raise RuntimeError("NEWSAPI_KEY is not set")

    resp = requests.get(
        NEWSAPI_URL,
        params={
            "q": UNIVERSE[ticker][0],
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": key,
        },
        timeout=10,
    )
    resp.raise_for_status()

    docs = []
    for a in resp.json().get("articles", []):
        text = ". ".join(p for p in (a.get("title"), a.get("description")) if p)
        if not text:
            continue
        docs.append(Document(
            source="newsapi", source_type="news",
            timestamp=a["publishedAt"], text=text, ticker=ticker,
        ))
    return docs


def load_news_csv(path: Path = DEFAULT_CSV) -> list[Document]:
    """Load offline news. Expected columns: timestamp, text[, ticker]."""
    df = pd.read_csv(path)
    docs = []
    for row in df.itertuples():
        ticker = getattr(row, "ticker", None)
        if not isinstance(ticker, str):
            ticker = tag_ticker(row.text)
        docs.append(Document(
            source="news_csv", source_type="news",
            timestamp=row.timestamp, text=row.text, ticker=ticker,
        ))
    return docs


def load_news(tickers: Optional[Iterable[str]] = None) -> list[Document]:
    """Live NewsAPI if possible, otherwise the offline CSV."""
    docs: list[Document] = []
    try:
        for t in (tickers or UNIVERSE):
            docs.extend(fetch_newsapi(t))
    except (RuntimeError, requests.RequestException) as e:
        print(f"[news] live fetch failed ({e}); using offline CSV")
        return load_news_csv()
    return docs
