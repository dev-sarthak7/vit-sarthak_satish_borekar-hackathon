from ..ingestion.tickers import UNIVERSE

# The mock index: same stocks the NLP engine covers
INDEX_TICKERS = list(UNIVERSE)


def equal_weights() -> dict[str, float]:
    """Starting portfolio: every stock gets the same weight."""
    w = 1 / len(INDEX_TICKERS)
    return {t: w for t in INDEX_TICKERS}
