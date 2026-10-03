import argparse
import json
from pathlib import Path

from ..ingestion.news import load_news, load_news_csv
from ..ingestion.social import load_tweets_csv
from .pipeline import Signal, process_documents

SIGNALS_PATH = Path("data/signals.json")
TWEETS_SAMPLE = Path("data/sample_tweets.csv")


def write_signals(signals: list[Signal], path: Path = SIGNALS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([s.model_dump(mode="json") for s in signals], indent=2))


def read_signals(path: Path = SIGNALS_PATH) -> list[Signal]:
    return [Signal.model_validate(x) for x in json.loads(path.read_text())]


def run_engine(live: bool = False, path: Path = SIGNALS_PATH,
               tweets_path: Path = TWEETS_SAMPLE) -> list[Signal]:
    """Ingest news + tweets, score everything, and write signals.json."""
    docs = load_news() if live else load_news_csv()
    if tweets_path.exists():
        docs += load_tweets_csv(tweets_path)
    docs.sort(key=lambda d: d.timestamp)
    signals = process_documents(docs)
    write_signals(signals, path)
    return signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the risk engine and write signals.json")
    parser.add_argument("--live", action="store_true",
                        help="fetch live headlines from NewsAPI (uses your daily quota)")
    args = parser.parse_args()
    out = run_engine(live=args.live)
    print(f"wrote {len(out)} signals to {SIGNALS_PATH}")
