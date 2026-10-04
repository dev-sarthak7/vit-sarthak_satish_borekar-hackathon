from pathlib import Path

import numpy as np
import pandas as pd

from ..ingestion.tickers import UNIVERSE

PORTFOLIO_PATH = Path("data/portfolio.csv")
SEED = 42

# Illustrative assumptions (not market data)
RATINGS = ["AA", "A", "BBB", "BB", "B"]
SPREAD = {"AAA": 0.0, "AA": 0.006, "A": 0.009, "BBB": 0.014, "BB": 0.030, "B": 0.050}
PD_PCT = {"AAA": 0.01, "AA": 0.03, "A": 0.06, "BBB": 0.20, "BB": 1.00, "B": 4.00}
LGD_PCT = 45.0   # senior unsecured, as in the Basel foundation approach

SECTOR = {
    "AAPL": "Technology", "MSFT": "Technology", "AMZN": "Consumer", "GOOG": "Technology",
    "META": "Technology", "TSLA": "Automotive", "AMD": "Technology", "NFLX": "Media",
    "PYPL": "Financials", "INTC": "Technology", "CRM": "Technology", "DIS": "Media",
    "BA": "Industrials", "PG": "Staples", "KO": "Staples", "COST": "Retail",
}
# Fictional borrowers so the loan book is not only our 16 companies
GENERIC_OBLIGORS = [
    ("Northern Utilities Ltd", "Utilities"), ("Atlas Energy Corp", "Energy"),
    ("Meridian Realty Trust", "Real Estate"), ("Harbor Shipping Co", "Industrials"),
    ("Summit Healthcare Inc", "Healthcare"), ("Crestline Telecom", "Telecom"),
    ("Pioneer Steel Works", "Materials"), ("Coastal Foods Group", "Staples"),
    ("Vertex Logistics", "Industrials"),
]

COLUMNS = [
    "position_id", "asset_class", "name", "ticker", "sector", "rating",
    "notional_musd", "market_value_musd", "maturity_years", "coupon_pct",
    "mod_duration", "convexity", "pd_pct", "lgd_pct", "direction",
]


def base_yield(years: float) -> float:
    """Illustrative risk-free yield curve."""
    return float(np.interp(years, [1, 5, 10, 30], [0.037, 0.039, 0.042, 0.045]))


def bond_metrics(coupon: float, years: int, y: float) -> dict:
    """Price (per 100), modified duration and convexity of an annual-coupon bond."""
    t = np.arange(1, years + 1)
    cf = np.full(years, 100 * coupon)
    cf[-1] += 100
    price = float((cf / (1 + y) ** t).sum())
    mod_dur = float((t * cf / (1 + y) ** (t + 1)).sum() / price)
    convexity = float((t * (t + 1) * cf / (1 + y) ** (t + 2)).sum() / price)
    return {"price": price, "mod_duration": mod_dur, "convexity": convexity}


def generate_portfolio(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    # Corporate loans: every universe company plus fictional borrowers
    obligors = [(UNIVERSE[t][0], t, SECTOR[t]) for t in UNIVERSE]
    obligors += [(n, None, s) for n, s in GENERIC_OBLIGORS]
    for i, (name, ticker, sector) in enumerate(obligors, 1):
        rating = str(rng.choice(RATINGS, p=[0.05, 0.25, 0.45, 0.20, 0.05]))
        exposure = round(float(rng.uniform(20, 200)), 1)
        years = int(rng.integers(1, 8))
        rows.append({
            "position_id": f"L{i:03d}", "asset_class": "loan",
            "name": f"{name} term loan", "ticker": ticker, "sector": sector,
            "rating": rating, "notional_musd": exposure, "market_value_musd": exposure,
            "maturity_years": years,
            "coupon_pct": round(100 * (base_yield(years) + SPREAD[rating] + 0.01), 2),
            "pd_pct": PD_PCT[rating], "lgd_pct": LGD_PCT,
        })

    # Corporate bonds
    for i, t in enumerate(rng.choice(list(UNIVERSE), size=10, replace=False), 1):
        rating = str(rng.choice(["A", "BBB", "BB"], p=[0.3, 0.5, 0.2]))
        years = int(rng.choice([3, 5, 7, 10]))
        y = base_yield(years) + SPREAD[rating]
        coupon = max(round(y + float(rng.normal(0, 0.004)), 4), 0.005)
        m = bond_metrics(coupon, years, y)
        notional = round(float(rng.uniform(10, 60)), 1)
        rows.append({
            "position_id": f"B{i:03d}", "asset_class": "corp_bond",
            "name": f"{UNIVERSE[str(t)][0]} {years}y bond", "ticker": str(t),
            "sector": SECTOR[str(t)], "rating": rating, "notional_musd": notional,
            "market_value_musd": round(notional * m["price"] / 100, 2),
            "maturity_years": years, "coupon_pct": round(100 * coupon, 2),
            "mod_duration": round(m["mod_duration"], 3),
            "convexity": round(m["convexity"], 2),
        })

    # Government bonds
    for i, years in enumerate([2, 5, 10, 10, 30], 1):
        y = base_yield(years)
        coupon = max(round(y + float(rng.normal(0, 0.003)), 4), 0.005)
        m = bond_metrics(coupon, years, y)
        notional = round(float(rng.uniform(50, 300)), 1)
        rows.append({
            "position_id": f"G{i:03d}", "asset_class": "gov_bond",
            "name": f"US Treasury {years}y", "sector": "Government", "rating": "AAA",
            "notional_musd": notional,
            "market_value_musd": round(notional * m["price"] / 100, 2),
            "maturity_years": years, "coupon_pct": round(100 * coupon, 2),
            "mod_duration": round(m["mod_duration"], 3),
            "convexity": round(m["convexity"], 2),
        })

    # Equity holdings
    for i, t in enumerate(rng.choice(list(UNIVERSE), size=10, replace=False), 1):
        value = round(float(rng.uniform(5, 60)), 1)
        rows.append({
            "position_id": f"E{i:03d}", "asset_class": "equity",
            "name": f"{UNIVERSE[str(t)][0]} shares", "ticker": str(t),
            "sector": SECTOR[str(t)], "notional_musd": value, "market_value_musd": value,
        })

    # Pay-fixed interest-rate swaps (gain when rates rise)
    for i in range(1, 7):
        years = int(rng.choice([3, 5, 7, 10]))
        y = base_yield(years)
        notional = float(rng.integers(10, 51) * 10)
        rows.append({
            "position_id": f"S{i:03d}", "asset_class": "ir_swap",
            "name": f"Pay-fixed IRS {years}y", "sector": "Rates", "notional_musd": notional,
            "market_value_musd": round(notional * float(rng.normal(0, 0.004)), 2),
            "maturity_years": years,
            "mod_duration": round((1 - (1 + y) ** -years) / y, 3),
            "direction": "pay_fixed",
        })

    # FX forwards (long USD gains when the dollar strengthens)
    for i, pair in enumerate(["EURUSD", "GBPUSD", "USDJPY", "USDCHF"], 1):
        notional = round(float(rng.uniform(50, 300)), 1)
        rows.append({
            "position_id": f"F{i:03d}", "asset_class": "fx_forward",
            "name": f"{pair} forward", "sector": "FX", "notional_musd": notional,
            "market_value_musd": round(notional * float(rng.normal(0, 0.004)), 2),
            "direction": str(rng.choice(["long_usd", "short_usd"])),
        })

    return pd.DataFrame(rows, columns=COLUMNS)


def write_portfolio(df: pd.DataFrame, path: Path = PORTFOLIO_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_portfolio(path: Path = PORTFOLIO_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    df = generate_portfolio()
    write_portfolio(df)
    summary = df.groupby("asset_class").agg(
        positions=("position_id", "count"),
        notional_musd=("notional_musd", "sum"),
        market_value_musd=("market_value_musd", "sum"),
    ).round(1)
    print(summary.to_string())
    print(f"\ntotal market value: {df['market_value_musd'].sum():.1f} USD m "
          f"across {len(df)} positions")
