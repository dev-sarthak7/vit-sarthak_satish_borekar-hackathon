import sys
from pathlib import Path

# Let "python -m"-style imports work when launched with `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.engine.export import read_signals
from src.rebalancer.backtest import excess_stats, performance, run_backtest
from src.rebalancer.prices import load_prices
from src.rebalancer.replay import weight_history
from src.stress.engine import run_stress, summarize
from src.stress.portfolio import load_portfolio
from src.stress.scenarios import SYSTEMIC, scaled_shock
from src.stress.trigger import triggered_signals

st.set_page_config(page_title="AI/NLP Risk Engine", layout="wide")
st.title("AI/NLP Risk Engine Dashboard")
st.caption(
    "Structured risk signals from news and social media drive a tactical index "
    "rebalancer (Module A) and an event-driven portfolio stress test (Module B)."
)

ASSET_NAMES = {
    "loan": "Loans (extra credit loss)", "corp_bond": "Corporate bonds",
    "gov_bond": "Government bonds", "equity": "Equities",
    "ir_swap": "Interest-rate swaps", "fx_forward": "FX forwards",
}


@st.cache_data
def get_signals():
    return read_signals()


@st.cache_data
def load_signals_df() -> pd.DataFrame:
    df = pd.DataFrame([s.model_dump() for s in get_signals()])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp", ascending=False)


@st.cache_data
def build_history() -> pd.DataFrame:
    return weight_history(get_signals(), load_prices().index)


@st.cache_data
def backtest_for(tilt: float) -> pd.DataFrame:
    daily, _ = run_backtest(get_signals(), load_prices(), tilt)
    return daily


@st.cache_data
def get_portfolio() -> pd.DataFrame:
    return load_portfolio()


tab_rebalancer, tab_stress, tab_feed = st.tabs(
    ["Index rebalancer", "Stress test", "Signal feed"])

# ================= Module A: index rebalancer =================
with tab_rebalancer:
    st.subheader("Index weights over time")
    st.caption("Historical replay: Sep 2021 - Sep 2022, using the tweets dataset.")
    weights = build_history()

    selected = st.multiselect("Stocks to show", list(weights.columns),
                              default=list(weights.columns))
    if selected:
        chart = weights[selected] * 100
        fig = px.line(chart, labels={"value": "Weight (%)", "date": "Date", "variable": "Stock"})
        fig.add_hline(y=100 / len(weights.columns), line_dash="dot",
                      annotation_text="equal weight")
        fig.update_layout(height=480, hovermode="x unified")
        st.plotly_chart(fig)
    else:
        st.info("Pick at least one stock.")

    st.subheader("Backtest vs equal-weight benchmark")
    tilt = st.select_slider(
        "Tilt strength (how strongly sentiment moves the weights)",
        options=[0.25, 0.5, 1.0], value=0.5,
    )
    daily = backtest_for(tilt)

    cumulative = ((1 + daily).cumprod() - 1) * 100
    cumulative = cumulative.rename(columns={"sentiment_index": "Sentiment index",
                                            "equal_weight": "Equal weight"})
    fig2 = px.line(cumulative, labels={"value": "Cumulative return (%)",
                                       "date": "Date", "variable": ""})
    fig2.update_layout(height=380, hovermode="x unified")
    st.plotly_chart(fig2)

    metrics = pd.DataFrame({
        "Sentiment index": performance(daily["sentiment_index"]),
        "Equal weight": performance(daily["equal_weight"]),
    }).round(2)
    st.dataframe(metrics)

    stats = excess_stats(daily)
    b1, b2, b3 = st.columns(3)
    b1.metric("Excess return (per year)", f"{stats['excess_return_%/yr']:+.2f}%")
    b2.metric("Tracking error (per year)", f"{stats['tracking_error_%/yr']:.2f}%")
    b3.metric("t-statistic", f"{stats['t_stat']:+.2f}")
    if abs(stats["t_stat"]) < 2:
        st.info("The difference from equal weight is not statistically distinguishable "
                "from noise (|t| < 2).")
    else:
        st.success("The difference is unlikely to be pure noise in this sample (|t| >= 2), "
                   "but this is a single year of data.")
    st.caption(
        "Weights from signals up to day t are applied to day t+1 returns (no look-ahead). "
        "One year (2021-22, a tech bear market); results are descriptive and make no claim "
        "of predictive power."
    )

# ================= Module B: stress test =================
with tab_stress:
    st.subheader("Event-driven portfolio stress test")
    st.caption(
        "Synthetic wholesale-banking book (loans, bonds, equities, swaps, FX forwards). "
        "A high-impact systemic event fires a scenario whose shocks scale with the "
        "event's impact score."
    )

    signals_list = get_signals()
    mode = st.radio("Scenario source", ["Triggered by a signal", "Custom scenario"],
                    horizontal=True)

    shock, scenario_label = None, ""
    if mode == "Triggered by a signal":
        t1, t2, t3 = st.columns(3)
        threshold = t1.slider("Impact threshold", 1.0, 10.0, 7.0, 0.5)
        news_only = t2.checkbox("News only", value=True)
        require_negative = t3.checkbox("Require negative sentiment", value=True)
        hits = triggered_signals(signals_list, threshold, require_negative, news_only)
        if not hits:
            st.info("No signal meets the trigger rule. Lower the threshold or include social posts.")
        else:
            k = st.selectbox(
                "Triggering signal", range(len(hits)),
                format_func=lambda i: (f"{hits[i].impact:.1f} | {hits[i].event} | "
                                       f"{hits[i].ticker} | {hits[i].timestamp:%Y-%m-%d} | "
                                       f"{hits[i].text[:70]}"),
            )
            sig = hits[k]
            shock = scaled_shock(sig.event, sig.impact)
            scenario_label = f"{sig.event} (impact {sig.impact:.1f})"
    else:
        c1, c2 = st.columns(2)
        event = c1.selectbox("Event type", list(SYSTEMIC))
        impact = c2.slider("Impact score", 1.0, 10.0, 8.0, 0.1)
        shock = scaled_shock(event, impact)
        scenario_label = f"{event} (impact {impact:.1f})"

    if shock is not None:
        st.markdown(f"**Scenario:** {scenario_label}")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Equities", f"{shock.equity_pct:+.1f}%")
        s2.metric("Risk-free yields", f"{shock.rate_bp:+.0f} bp")
        s3.metric("Credit spreads", f"{shock.spread_bp:+.0f} bp")
        s4.metric("US dollar", f"{shock.usd_pct:+.1f}%")
        s5.metric("Default probability", f"x{shock.pd_multiplier:.2f}")

        result = run_stress(get_portfolio(), shock)
        summary = summarize(result)
        total = summary.loc["TOTAL"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Portfolio value before", f"${total['value_before']:,.1f}M")
        m2.metric("Portfolio value after", f"${total['value_after']:,.1f}M",
                  f"{total['change']:+,.1f}M ({total['change_%']:+.2f}%)")
        m3.metric("Extra credit losses on loans", f"${-summary.loc['loan', 'change']:,.1f}M")

        by_class = summary.drop("TOTAL").reset_index()
        by_class["asset_class"] = by_class["asset_class"].map(ASSET_NAMES)
        by_class["effect"] = np.where(by_class["change"] >= 0, "Gain", "Loss")
        fig3 = px.bar(
            by_class, x="asset_class", y="change", color="effect",
            color_discrete_map={"Gain": "#16a34a", "Loss": "#dc2626"},
            labels={"asset_class": "", "change": "Change in value (USD m)", "effect": ""},
        )
        fig3.update_layout(height=380)
        st.plotly_chart(fig3)

        st.markdown("**Value by asset class (USD m)**")
        st.dataframe(summary.rename(index=ASSET_NAMES).round(2))

        st.markdown("**Most affected positions**")
        order = result["change"].abs().sort_values(ascending=False).index
        st.dataframe(
            result.loc[order].head(10)[["name", "asset_class", "value_before",
                                        "change", "value_after"]].round(2),
            hide_index=True,
        )

        with st.expander("How each asset class reacts"):
            st.markdown(
                "- **Equities:** value changes by the equity shock.\n"
                "- **Bonds:** duration-convexity approximation. Corporate bonds feel the "
                "rate shock plus the spread shock; government bonds the rate shock only.\n"
                "- **Loans:** held at amortised cost, so the stress shows up as extra "
                "expected credit loss: exposure x (stressed PD - PD) x LGD.\n"
                "- **Pay-fixed swaps:** gain when rates rise (notional x duration x rate move).\n"
                "- **FX forwards:** notional x dollar move, positive for long USD."
            )
        st.caption(
            "Shock sizes are illustrative assumptions, not calibrated to market data. "
            "Derivatives are valued without counterparty credit risk."
        )

# ================= Signal feed =================
with tab_feed:
    st.subheader("Signal feed")
    signals = load_signals_df()

    c1, c2, c3 = st.columns(3)
    tickers = c1.multiselect("Ticker", sorted(signals["ticker"].dropna().unique()))
    events = c2.multiselect("Event type", sorted(signals["event"].unique()))
    min_impact = c3.slider("Minimum impact", 1.0, 10.0, 1.0, 0.5)

    view = signals
    if tickers:
        view = view[view["ticker"].isin(tickers)]
    if events:
        view = view[view["event"].isin(events)]
    view = view[view["impact"] >= min_impact]

    m1, m2, m3 = st.columns(3)
    m1.metric("Signals shown", len(view))
    m2.metric("Average sentiment",
              f"{view['sentiment'].mean():+.2f}" if len(view) else "n/a")
    m3.metric("High impact (7+)", int((view["impact"] >= 7).sum()))

    st.dataframe(
        view[["timestamp", "ticker", "source_type", "sentiment", "event", "impact", "text"]].head(200),
        hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
            "ticker": "Ticker",
            "source_type": "Source",
            "sentiment": st.column_config.NumberColumn("Sentiment", format="%+.2f"),
            "event": "Event",
            "impact": st.column_config.ProgressColumn("Impact", min_value=1, max_value=10, format="%.1f"),
            "text": st.column_config.TextColumn("Text", width="large"),
        },
    )
    st.caption("Showing the 200 most recent matching signals.")
