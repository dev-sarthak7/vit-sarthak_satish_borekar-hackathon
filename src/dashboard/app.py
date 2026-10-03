import sys
from pathlib import Path

# Let "python -m"-style imports work when launched with `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px
import streamlit as st

from src.engine.export import read_signals
from src.rebalancer.prices import load_prices
from src.rebalancer.replay import weight_history

st.set_page_config(page_title="Sentiment Index Rebalancer", layout="wide")
st.title("Tactical Index Rebalancer")
st.caption(
    "Index weights react to sentiment signals from the NLP Risk Engine. "
    "Historical replay: Sep 2021 - Sep 2022, using the tweets dataset."
)


@st.cache_data
def build_history():
    signals = read_signals()
    prices = load_prices()
    return weight_history(signals, prices.index)


weights = build_history()

selected = st.multiselect("Stocks to show", list(weights.columns),
                          default=list(weights.columns))
if not selected:
    st.info("Pick at least one stock.")
    st.stop()

chart = weights[selected] * 100
fig = px.line(chart, labels={"value": "Weight (%)", "date": "Date", "variable": "Stock"})
fig.add_hline(y=100 / len(weights.columns), line_dash="dot",
              annotation_text="equal weight")
fig.update_layout(height=520, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)
