from pathlib import Path

import pandas as pd

from .events import classify_event
from .events_zeroshot import classify_event_zeroshot
from .pipeline import _resolve_event

EVAL_PATH = Path("data/event_eval.csv")


def evaluate(path: Path = EVAL_PATH) -> pd.DataFrame:
    """Predictions of the three systems for every labeled headline."""
    df = pd.read_csv(path)
    df["keyword"] = df["text"].map(classify_event)
    df["zero_shot"] = df["text"].map(lambda t: classify_event_zeroshot(t)[0])
    df["final"] = df["text"].map(lambda t: _resolve_event(t)[0])
    return df


if __name__ == "__main__":
    df = evaluate()
    n = len(df)
    for col in ("keyword", "zero_shot", "final"):
        correct = int((df[col] == df["label"]).sum())
        print(f"{col:10s} accuracy {100 * correct / n:5.1f}%  ({correct}/{n})")
    wrong = df[df["final"] != df["label"]]
    print(f"\nFinal system errors ({len(wrong)}):")
    for r in wrong.itertuples():
        print(f"  gold={r.label:18s} pred={r.final:18s} | {r.text[:70]}")
