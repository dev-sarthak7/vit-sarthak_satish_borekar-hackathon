import pandas as pd

from .scenarios import Shock

# Derivatives have market values near zero, so a % change is meaningless for them
DERIVATIVES = ("ir_swap", "fx_forward")


def _bond_change(row: pd.Series, dy: float) -> float:
    """Duration-convexity approximation of a bond's value change for a yield change dy."""
    return row["market_value_musd"] * (
        -row["mod_duration"] * dy + 0.5 * row["convexity"] * dy ** 2
    )


def position_change(row: pd.Series, shock: Shock) -> float:
    """Change in value (USD millions) of one position under a shock."""
    asset = row["asset_class"]

    if asset == "equity":
        return row["market_value_musd"] * shock.equity_pct / 100
    if asset == "gov_bond":
        return _bond_change(row, shock.rate_bp / 10_000)
    if asset == "corp_bond":
        return _bond_change(row, (shock.rate_bp + shock.spread_bp) / 10_000)
    if asset == "loan":
        # Held at amortised cost: the stress appears as extra expected credit loss
        pd_before = row["pd_pct"] / 100
        pd_after = min(pd_before * shock.pd_multiplier, 1.0)
        return -row["notional_musd"] * (pd_after - pd_before) * row["lgd_pct"] / 100
    if asset == "ir_swap":      # pay-fixed: gains when rates rise
        return row["notional_musd"] * row["mod_duration"] * shock.rate_bp / 10_000
    if asset == "fx_forward":
        sign = 1 if row["direction"] == "long_usd" else -1
        return row["notional_musd"] * shock.usd_pct / 100 * sign
    raise ValueError(f"unknown asset class: {asset}")


def run_stress(portfolio: pd.DataFrame, shock: Shock) -> pd.DataFrame:
    """Per-position value before, change and value after a shock."""
    out = portfolio[["position_id", "asset_class", "name", "ticker",
                     "market_value_musd"]].rename(
        columns={"market_value_musd": "value_before"})
    out["change"] = portfolio.apply(lambda r: position_change(r, shock), axis=1)
    out["value_after"] = out["value_before"] + out["change"]
    return out


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    """Totals by asset class plus a TOTAL row."""
    g = result.groupby("asset_class")[["value_before", "change", "value_after"]].sum()
    g.loc["TOTAL"] = g.sum()
    g["change_%"] = 100 * g["change"] / g["value_before"]
    g.loc[list(DERIVATIVES), "change_%"] = float("nan")
    return g
