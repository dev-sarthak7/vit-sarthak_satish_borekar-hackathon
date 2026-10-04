import numpy as np
import pandas as pd

from ..engine.pipeline import Signal
from .replay import weight_history
from .weights import TILT

TRADING_DAYS = 252


def run_backtest(signals: list[Signal], prices: pd.DataFrame, tilt: float = TILT):
    """Daily returns of the sentiment index vs an equal-weight index.

    Weights computed from signals up to the end of day t are applied to the
    returns of day t+1, so there is no look-ahead.
    Returns (daily_returns DataFrame, weights DataFrame).
    """
    returns = prices.pct_change().dropna()
    weights = weight_history(signals, prices.index, tilt=tilt)
    applied = weights.shift(1).reindex(returns.index)
    daily = pd.DataFrame({
        "sentiment_index": (applied * returns).sum(axis=1),
        "equal_weight": returns.mean(axis=1),
    })
    return daily, weights


def performance(daily: pd.Series) -> dict:
    cumulative = (1 + daily).cumprod()
    std = daily.std()
    return {
        "total_return_%": 100 * (cumulative.iloc[-1] - 1),
        "ann_vol_%": 100 * std * np.sqrt(TRADING_DAYS),
        "sharpe": daily.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else float("nan"),
        "max_drawdown_%": 100 * (cumulative / cumulative.cummax() - 1).min(),
    }


def excess_stats(daily: pd.DataFrame) -> dict:
    """Is the sentiment index's edge over the benchmark distinguishable from noise?"""
    diff = daily["sentiment_index"] - daily["equal_weight"]
    std = diff.std()
    t = diff.mean() / (std / np.sqrt(len(diff))) if std > 1e-10 else float("nan")
    return {
        "excess_return_%/yr": 100 * diff.mean() * TRADING_DAYS,
        "tracking_error_%/yr": 100 * std * np.sqrt(TRADING_DAYS),
        "t_stat": t,
    }


def turnover(weights: pd.DataFrame) -> float:
    """Average share of the portfolio traded per day, in %."""
    return 100 * weights.diff().abs().sum(axis=1).mean() / 2


if __name__ == "__main__":
    from ..engine.export import read_signals
    from .prices import load_prices

    signals, prices = read_signals(), load_prices()
    rows, bench = [], None
    for tilt in (0.0, 0.25, 0.5, 1.0):
        daily, weights = run_backtest(signals, prices, tilt)
        rows.append({"tilt": tilt, **performance(daily["sentiment_index"]),
                     **excess_stats(daily), "turnover_%/day": turnover(weights)})
        bench = performance(daily["equal_weight"])
    print("Sentiment index by tilt strength:")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nEqual-weight benchmark:")
    print(pd.Series(bench).round(2).to_string())
