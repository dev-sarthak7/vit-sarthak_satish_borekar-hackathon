from src.stress.scenarios import SCENARIOS, scaled_shock, severity_scale


def test_severity_scale_is_clamped():
    assert severity_scale(3.5) == 0.5       # floor
    assert severity_scale(7.0) == 1.0       # reference impact
    assert severity_scale(14.0) == 1.5      # cap


def test_scaled_shock_at_reference_impact_equals_base():
    assert scaled_shock("Credit Event", 7.0) == SCENARIOS["Credit Event"]


def test_scaled_shock_scales_linearly():
    s = scaled_shock("Credit Event", 10.5)      # scale 1.5
    assert s.equity_pct == -18.0
    assert s.spread_bp == 225.0
    assert abs(s.pd_multiplier - 2.2) < 1e-9


def test_company_specific_events_have_no_scenario():
    assert scaled_shock("Earnings", 9.0) is None
    assert scaled_shock("Other", 9.0) is None


def test_geopolitical_is_flight_to_quality():
    g = SCENARIOS["Geopolitical"]
    assert g.rate_bp < 0 and g.spread_bp > 0 and g.usd_pct > 0
