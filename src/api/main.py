from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from ..engine.export import SIGNALS_PATH, read_signals
from ..engine.pipeline import Signal
from ..ingestion.tickers import UNIVERSE

app = FastAPI(title="AI/NLP Risk Engine", version="0.1.0")


def _load() -> list[Signal]:
    if not SIGNALS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="signals.json not found. Run: python -m src.engine.export",
        )
    return read_signals()


@app.get("/health")
def health():
    return {"status": "ok", "signals_file": SIGNALS_PATH.exists()}


@app.get("/signals", response_model=list[Signal])
def get_signals(
    ticker: Optional[str] = None,
    event: Optional[str] = None,
    min_impact: float = Query(1.0, ge=1.0, le=10.0),
    limit: int = Query(100, ge=1, le=1000),
):
    """Latest signals first, with optional filters."""
    signals = _load()
    if ticker:
        signals = [s for s in signals if s.ticker == ticker.upper()]
    if event:
        signals = [s for s in signals if s.event.lower() == event.lower()]
    signals = [s for s in signals if s.impact >= min_impact]
    signals.sort(key=lambda s: s.timestamp, reverse=True)
    return signals[:limit]


@app.get("/signals/{ticker}", response_model=list[Signal])
def get_ticker_signals(ticker: str, limit: int = Query(100, ge=1, le=1000)):
    t = ticker.upper()
    if t not in UNIVERSE:
        raise HTTPException(status_code=404, detail=f"{t} is not in the index universe")
    signals = [s for s in _load() if s.ticker == t]
    signals.sort(key=lambda s: s.timestamp, reverse=True)
    return signals[:limit]
