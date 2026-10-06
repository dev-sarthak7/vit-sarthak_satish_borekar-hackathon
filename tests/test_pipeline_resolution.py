from src.engine import pipeline


def test_systemic_label_without_keyword_support_falls_back_to_keywords(monkeypatch):
    monkeypatch.setattr(pipeline, "classify_event_zeroshot", lambda t: ("Credit Event", 0.95))
    label, _ = pipeline._resolve_event("Tech shares fall 22% from January peak amid broad sell-off")
    assert label == "Market Move"


def test_systemic_label_with_keyword_support_is_kept(monkeypatch):
    monkeypatch.setattr(pipeline, "classify_event_zeroshot", lambda t: ("Credit Event", 0.95))
    label, conf = pipeline._resolve_event("Rating agency downgrades chipmaker to BBB- citing heavy debt load")
    assert (label, conf) == ("Credit Event", 0.95)


def test_weak_guess_with_no_keywords_becomes_other(monkeypatch):
    monkeypatch.setattr(pipeline, "classify_event_zeroshot", lambda t: ("Earnings", 0.68))
    label, _ = pipeline._resolve_event("Company will hold its annual shareholder meeting on Tuesday")
    assert label == "Other"
