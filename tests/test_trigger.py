from src.engine.pipeline import Signal
from src.stress.trigger import triggered_signals, triggers_stress


def make(event="Credit Event", impact=9.7, sentiment=-0.96) -> Signal:
    return Signal(timestamp="2026-09-29T08:45:00Z", ticker="INTC", source="x",
                  source_type="news", text="headline", sentiment=sentiment,
                  event=event, event_confidence=0.9, impact=impact)


def test_fires_on_severe_negative_systemic_event():
    assert triggers_stress(make())


def test_ignores_company_specific_events():
    assert not triggers_stress(make(event="Earnings", impact=9.0))


def test_ignores_low_impact_events():
    assert not triggers_stress(make(event="Geopolitical", impact=6.9))


def test_positive_sentiment_does_not_trigger_by_default():
    positive = make(event="Geopolitical", impact=8.0, sentiment=0.7)
    assert not triggers_stress(positive)
    assert triggers_stress(positive, require_negative=False)


def test_triggered_signals_are_sorted_by_impact():
    hits = triggered_signals([make(impact=8.1), make(impact=9.7), make(impact=7.5)])
    assert [s.impact for s in hits] == [9.7, 8.1, 7.5]


def test_social_posts_do_not_trigger_by_default():
    tweet = make().model_copy(update={"source_type": "social"})
    assert not triggers_stress(tweet)
    assert triggers_stress(tweet, news_only=False)
