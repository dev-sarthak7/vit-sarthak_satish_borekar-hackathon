import re
from typing import Optional

# Mock index universe: ticker -> names the text may use for the company
UNIVERSE = {
    "AAPL": ["Apple"],
    "MSFT": ["Microsoft"],
    "AMZN": ["Amazon"],
    "GOOGL": ["Google", "Alphabet"],
    "META": ["Meta Platforms", "Facebook"],
    "NVDA": ["Nvidia"],
    "TSLA": ["Tesla"],
    "JPM": ["JPMorgan", "JP Morgan"],
    "BAC": ["Bank of America"],
    "GS": ["Goldman Sachs"],
    "XOM": ["Exxon", "ExxonMobil"],
    "JNJ": ["Johnson & Johnson"],
    "PG": ["Procter & Gamble"],
    "WMT": ["Walmart"],
    "KO": ["Coca-Cola", "Coca Cola"],
}


def tag_ticker(text: str) -> Optional[str]:
    """Return the first universe ticker mentioned in text (by name or $CASHTAG)."""
    for ticker, names in UNIVERSE.items():
        if re.search(rf"\${ticker}\b", text, re.IGNORECASE):
            return ticker
        for name in names:
            if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
                return ticker
    return None
