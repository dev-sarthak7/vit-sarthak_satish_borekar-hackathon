import re
from typing import Optional

# Mock index universe: ticker -> names the text may use for the company.
# Chosen because the tweets dataset has solid labeled coverage for each.
UNIVERSE = {
    "AAPL": ["Apple"],
    "MSFT": ["Microsoft"],
    "AMZN": ["Amazon"],
    "GOOG": ["Google", "Alphabet"],
    "META": ["Meta Platforms", "Facebook"],
    "TSLA": ["Tesla"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "NFLX": ["Netflix"],
    "PYPL": ["PayPal"],
    "INTC": ["Intel"],
    "CRM": ["Salesforce"],
    "DIS": ["Disney"],
    "BA": ["Boeing"],
    "PG": ["Procter & Gamble"],
    "KO": ["Coca-Cola", "Coca Cola"],
    "COST": ["Costco"],
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
