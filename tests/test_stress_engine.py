import math

import pandas as pd
import pytest

from src.stress.engine import run_stress, summarize
from src.stress.scenarios import SCENARIOS

GEO = SCENARIOS["Geopolitical"]   # equity -10%, rates -30bp, spread +80bp, USD +3%, PD x1.3


def small_portfolio(pd_pct: float = 2.0) -> pd.DataFrame:
    base = dict(ticker=None, name="x", convexity=None, mod_duration=None,
                pd_pct=None, lgd_pct=None, direction=None, notional_musd=None)
    rows = [
        dict(position_id="E", asset_class="equity", market_value_musd=100, **base),
        dict(position_id="G", asset_class="gov_bond", market_value_musd=100,
             **{**base, "mod_duration": 8, "convexity": 80}),
        dict(position_id="B", asset_class="corp_bond", market_value_musd=100,
             **{**base, "mod_duration": 5, "convexity": 30}),
        dict(position_id="L", asset_class="loan", market_value_musd=100,
             **{**base, "notional_musd": 100, "pd_pct": pd_pct, "lgd_pct": 45.0}),
        dict(position_id="S", asset_class="ir_swap", market_value_musd=0,
             **{**base, "notional_musd": 200, "mod_duration": 5}),
        dict(position_id="FL", asset_class="fx_forward", market_value_musd=0,
             **{**base, "notional_musd": 100, "direction": "long_usd"}),
        dict(position_id="FS", asset_class="fx_forward", market_value_musd=0,
             **{**base, "notional_musd": 100, "direction": "short_usd"}),
    ]
    return pd.DataFrame(rows)


def changes(result: pd.DataFrame) -> dict:
    return dict(zip(result["position_id"], result["change"]))


def test_position_changes_match_hand_calculations():
    c = changes(run_stress(small_portfolio(), GEO))
    assert c["E"] == pytest.approx(-10.0)
    assert c["G"] == pytest.approx(2.436)       # 100 * (-8*-0.003 + 0.5*80*0.003**2)
    assert c["B"] == pytest.approx(-2.4625)     # dy = +0.005: 100 * (-5*0.005 + 0.5*30*0.005**2)
    assert c["L"] == pytest.approx(-0.27)       # 100 * (0.026 - 0.020) * 0.45
    assert c["S"] == pytest.approx(-3.0)        # 200 * 5 * -0.003
    assert c["FL"] == pytest.approx(3.0)
    assert c["FS"] == pytest.approx(-3.0)


def test_total_change():
    result = run_stress(small_portfolio(), GEO)
    assert result["change"].sum() == pytest.approx(-13.2965)


def test_loan_default_probability_is_capped_at_100_percent():
    c = changes(run_stress(small_portfolio(pd_pct=90.0), GEO))
    assert c["L"] == pytest.approx(-4.5)        # PD 90% -> 100%: 100 * 0.10 * 0.45


def test_summary_has_total_and_blanks_derivative_percentages():
    summary = summarize(run_stress(small_portfolio(), GEO))
    assert summary.loc["TOTAL", "change"] == pytest.approx(-13.2965)
    assert math.isnan(summary.loc["ir_swap", "change_%"])
    assert math.isnan(summary.loc["fx_forward", "change_%"])
