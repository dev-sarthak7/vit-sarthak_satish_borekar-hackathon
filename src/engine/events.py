import re

# Order matters: it is the tie-break priority (earlier wins a tie).
EVENT_KEYWORDS = {
    "Credit Event": [
        r"defaults?", r"downgrad\w*", r"bankruptcy", r"insolven\w*",
        r"credit (?:rating|crunch|losses|risk)", r"debt restructuring",
        r"write-?downs?", r"loan losses", r"delinquen\w*", r"chapter 11",
        r"liquidity crisis",
    ],
    "Geopolitical": [
        r"war", r"invasion", r"sanctions?", r"tensions?", r"conflict",
        r"military", r"missile", r"ceasefire", r"embargo", r"tariffs?",
        r"geopolitic\w*", r"middle east", r"ukraine", r"russia", r"taiwan",
        r"coup", r"troops",
    ],
    "Macroeconomic": [
        r"inflation", r"interest rates?", r"rate (?:hike|hikes|cut|cuts)",
        r"federal reserve", r"fed", r"gdp", r"recession", r"unemployment",
        r"jobs report", r"cpi", r"central bank", r"treasury yields?",
        r"stimulus",
    ],
    "Merger/Acquisition": [
        r"acqui\w+", r"merger", r"merges?", r"takeover", r"buyout", r"to buy",
    ],
    "Regulatory/Legal": [
        r"probe", r"investigation", r"lawsuit", r"sued?", r"fined?",
        r"antitrust", r"regulators?", r"sec", r"settlement", r"bans?|banned",
    ],
    "Earnings": [
        r"earnings", r"revenue", r"guidance", r"quarterly", r"profits?",
        r"forecast", r"beats? (?:analyst )?expectations?", r"eps",
    ],
    "Product Launch": [
        r"unveil\w*", r"launch\w*", r"announces? new", r"introduc\w+",
        r"rolls? out", r"rollout", r"debut\w*", r"new iphone", r"releases?|released",
    ],
}

_COMPILED = {
    label: [re.compile(rf"\b(?:{p})\b", re.IGNORECASE) for p in patterns]
    for label, patterns in EVENT_KEYWORDS.items()
}


def event_scores(text: str) -> dict[str, int]:
    """Number of distinct keyword patterns matched per event type."""
    return {
        label: sum(1 for rx in rxs if rx.search(text))
        for label, rxs in _COMPILED.items()
    }


def classify_event(text: str) -> str:
    """Return the best-matching event type, or 'Other' if nothing matches."""
    scores = event_scores(text)
    best = max(scores.values())
    if best == 0:
        return "Other"
    # dict order = priority order, so the first label with the top score wins ties
    return next(label for label, s in scores.items() if s == best)
