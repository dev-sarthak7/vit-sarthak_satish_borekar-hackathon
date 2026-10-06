# Base market severity per event type (0-1). Judgment-based assumption.
SEVERITY = {
    "Credit Event": 1.00,
    "Geopolitical": 0.95,
    "Macroeconomic": 0.85,
    "Merger/Acquisition": 0.65,
    "Regulatory/Legal": 0.60,
    "Earnings": 0.55,
    "Product Launch": 0.35,
    "Market Move": 0.40,
    "Other": 0.15,
}

# Social media is noisier than news, so it counts for less.
SOURCE_WEIGHT = {"news": 1.0, "social": 0.7}


def impact_score(
    sentiment: float,
    event: str,
    event_conf: float = 1.0,
    source_type: str = "news",
) -> float:
    """Predicted market impact on a 1-10 scale, rounded to 1 decimal."""
    severity = SEVERITY.get(event, SEVERITY["Other"])
    source_w = SOURCE_WEIGHT.get(source_type, SOURCE_WEIGHT["social"])
    magnitude = 0.4 + 0.6 * min(abs(sentiment), 1.0)
    confidence = 0.6 + 0.4 * max(0.0, min(event_conf, 1.0))

    raw = severity * magnitude * source_w * confidence
    return round(1 + 9 * raw, 1)
