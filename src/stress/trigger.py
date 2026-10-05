from ..engine.pipeline import Signal
from .engine import run_stress
from .scenarios import SYSTEMIC, scaled_shock

DEFAULT_THRESHOLD = 7.0


def triggers_stress(signal: Signal, threshold: float = DEFAULT_THRESHOLD,
                    require_negative: bool = True, news_only: bool = True) -> bool:
    """A systemic event with high enough impact (and adverse sentiment) fires a stress test.

    By default only news can fire one: social posts are noisier and their event
    labels are less reliable.
    """
    if news_only and signal.source_type != "news":
        return False
    if signal.event not in SYSTEMIC:
        return False
    if signal.impact < threshold:
        return False
    if require_negative and signal.sentiment >= 0:
        return False
    return True


def triggered_signals(signals: list[Signal], threshold: float = DEFAULT_THRESHOLD,
                      require_negative: bool = True, news_only: bool = True) -> list[Signal]:
    """Signals that fire a stress test, most severe first."""
    hits = [s for s in signals
            if triggers_stress(s, threshold, require_negative, news_only)]
    return sorted(hits, key=lambda s: s.impact, reverse=True)


def stress_for_signal(portfolio, signal: Signal):
    """Run the scenario for a triggering signal. Returns (shock, per-position result)."""
    shock = scaled_shock(signal.event, signal.impact)
    return shock, run_stress(portfolio, shock)
