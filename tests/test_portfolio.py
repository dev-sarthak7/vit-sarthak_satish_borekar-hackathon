from src.ingestion.tickers import UNIVERSE
from src.stress.portfolio import bond_metrics, generate_portfolio


def test_generation_is_deterministic():
    assert generate_portfolio().equals(generate_portfolio())


def test_position_counts():
    counts = generate_portfolio()["asset_class"].value_counts().to_dict()
    assert counts == {"loan": 25, "corp_bond": 10, "equity": 10,
                      "ir_swap": 6, "gov_bond": 5, "fx_forward": 4}


def test_bond_metrics_for_par_bond():
    m = bond_metrics(coupon=0.04, years=10, y=0.04)
    assert abs(m["price"] - 100) < 1e-9
    assert abs(m["mod_duration"] - 8.1109) < 1e-3     # (1 - 1.04**-10) / 0.04


def test_loans_have_pd_and_lgd():
    loans = generate_portfolio().query("asset_class == 'loan'")
    assert (loans["pd_pct"] > 0).all()
    assert (loans["lgd_pct"] == 45.0).all()


def test_tickers_come_from_the_universe():
    tickers = generate_portfolio()["ticker"].dropna()
    assert set(tickers) <= set(UNIVERSE)
