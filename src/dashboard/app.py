import sys
from pathlib import Path

# Let "python -m"-style imports work when launched with `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.engine.export import read_signals
from src.rebalancer.backtest import excess_stats, performance, run_backtest
from src.rebalancer.prices import load_prices
from src.rebalancer.replay import weight_history

st.set_page_config(page_title="Sentiment Index Rebalancer", layout="wide")
st.title("Tactical Index Rebalancer")
st.caption(
    "Index weights react to sentiment signals from the NLP Risk Engine. "
    "Historical replay: Sep 2021 - Sep 2022, using the tweets dataset."
)


@st.cache_data
def load_signals_df() -> pd.DataFrame:
    df = pd.DataFrame([s.model_dump() for s in read_signals()])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp", ascending=False)


@st.cache_data
def build_history() -> pd.DataFrame:
    return weight_history(read_signals(), load_prices().index)


@st.cache_data
def backtest_for(tilt: float) -> pd.DataFrame:
    daily, _ = run_backtest(read_signals(), load_prices(), tilt)
    return daily


# ---- Section 1: index weights over time ----
st.subheader("Index weights over time")
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

# ---- Section 2: backtest against the equal-weight benchmark ----
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

# ---- Section 3: signal feed from the NLP engine ----
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
