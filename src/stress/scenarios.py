from dataclasses import dataclass


@dataclass(frozen=True)
class Shock:
    equity_pct: float      # % change in equity prices
    rate_bp: float         # parallel move in risk-free yields, basis points
    spread_bp: float       # widening of corporate credit spreads, basis points
    usd_pct: float         # % change of the US dollar against other currencies
    pd_multiplier: float   # multiplier on loan default probabilities


# Base shocks per systemic event type. Illustrative assumptions, not calibrated
# to market data; they are meant to be easy to read and easy to change.
SCENARIOS = {
    # Flight to quality: yields fall, spreads widen, the dollar strengthens
    "Geopolitical": Shock(equity_pct=-10, rate_bp=-30, spread_bp=80, usd_pct=3, pd_multiplier=1.30),
    # Tightening: yields rise, mild equity sell-off, moderate spread widening
    "Macroeconomic": Shock(equity_pct=-7, rate_bp=75, spread_bp=40, usd_pct=1, pd_multiplier=1.15),
    # Credit stress: sharp spread widening and a large rise in defaults
    "Credit Event": Shock(equity_pct=-12, rate_bp=-10, spread_bp=150, usd_pct=0, pd_multiplier=1.80),
}
SYSTEMIC = tuple(SCENARIOS)

REFERENCE_IMPACT = 7.0          # an impact of 7 applies the base shock as is
MIN_SCALE, MAX_SCALE = 0.5, 1.5


def severity_scale(impact: float) -> float:
    """Scale factor for a signal's impact score, kept between 0.5 and 1.5."""
    return max(MIN_SCALE, min(MAX_SCALE, impact / REFERENCE_IMPACT))


def scaled_shock(event: str, impact: float) -> Shock | None:
    """Shock for an event type scaled by impact, or None for non-systemic events."""
    base = SCENARIOS.get(event)
    if base is None:
        return None
    s = severity_scale(impact)
    return Shock(
        equity_pct=base.equity_pct * s,
        rate_bp=base.rate_bp * s,
        spread_bp=base.spread_bp * s,
        usd_pct=base.usd_pct * s,
        pd_multiplier=1 + (base.pd_multiplier - 1) * s,
    )
