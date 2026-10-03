import pytest
from pydantic import ValidationError

from src.ingestion.news import load_news, load_news_csv
from src.ingestion.schema import Document
from src.ingestion.social import load_tweets_csv
from src.ingestion.tickers import tag_ticker


def test_document_cleans_whitespace():
    d = Document(source="x", source_type="news",
                 timestamp="2026-09-28T09:00:00Z", text="  Tesla \n rises  ")
    assert d.text == "Tesla rises"


def test_document_rejects_empty_text():
    with pytest.raises(ValidationError):
        Document(source="x", source_type="news",
                 timestamp="2026-09-28T09:00:00Z", text="   ")


def test_document_rejects_bad_source_type():
    with pytest.raises(ValidationError):
        Document(source="x", source_type="blog",
                 timestamp="2026-09-28T09:00:00Z", text="hello")


def test_tag_ticker_by_name_and_cashtag():
    assert tag_ticker("Tesla recalls cars") == "TSLA"
    assert tag_ticker("buying $NVDA today") == "NVDA"
    assert tag_ticker("Weather is nice today") is None


def test_news_csv_loads_and_tags():
    docs = load_news_csv()
    assert len(docs) == 8
    assert all(d.source_type == "news" for d in docs)
    assert any(d.ticker == "TSLA" for d in docs)


def test_news_falls_back_to_csv_without_key(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "")
    docs = load_news(["TSLA"])
    assert len(docs) == 8
    assert docs[0].source == "news_csv"


def test_tweets_loader_filters_to_universe(tmp_path):
    f = tmp_path / "tweets.csv"
    f.write_text(
        "Date,Tweet,Stock Name\n"
        "2026-09-28,$TSLA to the moon,TSLA\n"
        "2026-09-28,random tweet about food,ZZZZ\n"
        "2026-09-28,Nvidia chips look strong,XYZ\n"
    )
    docs = load_tweets_csv(f)
    assert [d.ticker for d in docs] == ["TSLA", "NVDA"]
    assert all(d.source_type == "social" for d in docs)


def test_tweets_loader_maps_goog_alias(tmp_path):
    f = tmp_path / "tweets.csv"
    f.write_text("Date,Tweet,Stock Name\n2026-09-28,search ads are up,GOOG\n")
    docs = load_tweets_csv(f)
    assert [d.ticker for d in docs] == ["GOOGL"]


def test_tweets_loader_drops_bad_dates(tmp_path):
    f = tmp_path / "tweets.csv"
    f.write_text(
        "Date,Tweet,Stock Name\n"
        "not a date,$TSLA up,TSLA\n"
        "2026-09-28,$AAPL up,AAPL\n"
    )
    docs = load_tweets_csv(f)
    assert [d.ticker for d in docs] == ["AAPL"]


def test_sample_tweets_file_round_trips():
    docs = load_tweets_csv("data/sample_tweets.csv")
    assert len(docs) == 560
    assert len({d.ticker for d in docs}) == 15
