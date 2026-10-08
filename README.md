# AI/NLP Risk Engine + Tactical Rebalancer & Portfolio Stress Tester - S&P Global & Crisil Campus Hackathon

**Candidate Name:** Sarthak Satish Borekar  
**College Email ID:** sarthak.23bce10568@vitbhopal.ac.in  
**College / Campus:** Vellore Institute of Technology - Bhopal  
**Demo Video Link:** [YouTube / Unlisted - Video Walkthrough](https://youtu.be/placeholder) *(Update with unlisted demo link)*  
**Slide Deck Link (if hosted externally):** [`docs/presentation.pdf`](docs/presentation.pdf)

---

## 1. Project Overview / Problem Statement & Approach

Financial institutions, risk managers, and asset allocators are inundated with massive volumes of unstructured, high-velocity text from financial news wires, regulatory filings, and social media platforms. Sifting through this unstructured data manually to quantify market exposure in real-time is impossible. This project delivers an end-to-end platform centered around a unified **AI/NLP Risk Engine** that ingests multi-source text, extracts actionable quantitative signals, and feeds two complementary downstream applications: **Module A (Tactical Index Rebalancer)** and **Module B (Strategic Portfolio Stress Testing)**.

### Core: AI/NLP Risk Engine

The engine parses raw text across multiple channels (financial news and social media tweets) and produces standardized, machine-readable risk signals containing:

| Output Field | Definition & Scoring Logic | Numerical Range / Categories |
|---|---|---|
| **Sentiment Score** | Domain-specific sentiment polarity computed via **FinBERT** (`P(pos) - P(neg)`) | `[-1.0, +1.0]` (Negative to Positive) |
| **Event Classification** | 9-category classification using Zero-Shot NLI (`cross-encoder/nli-distilroberta-base`) backed by keyword verification | `Credit Event`, `Geopolitical`, `Macroeconomic`, `Merger/Acquisition`, `Regulatory/Legal`, `Earnings`, `Product Launch`, `Market Move`, `Other` |
| **Impact Score** | Severity formula modeling event magnitude, sentiment intensity, source credibility, and classification confidence | `[1.0, 10.0]` (1 = Minimal, 10 = Systemic shock) |

Structured signals are published to `data/signals.json` and served dynamically via a RESTful **FastAPI** service.

```
+---------------------------------------------------------------------------------------------+
|                                    DATA INGESTION                                           |
|   +---------------------------------------+     +---------------------------------------+   |
|   | Financial News (CSV / Live NewsAPI)   |     | Social Media (Stock Tweets Dataset)   |   |
|   +---------------------------------------+     +---------------------------------------+   |
+---------------------------------------------------------------------------------------------+
                                              |
                                              v
+---------------------------------------------------------------------------------------------+
|                                 CORE AI/NLP RISK ENGINE                                     |
|  +---------------------------+  +---------------------------+  +-------------------------+  |
|  |   FinBERT Sentiment       |  |  Zero-Shot NLI Classifier |  |  Severity-Based Impact  |  |
|  |   [-1.0 to +1.0]          |  |  9 Event Categories       |  |  [1.0 to 10.0]          |  |
|  +---------------------------+  +---------------------------+  +-------------------------+  |
|                                             |                                               |
|                    Structured Signal: (Ticker, Sentiment, Event, Impact)                    |
+---------------------------------------------------------------------------------------------+
                          |                                             |
                          v                                             v
+---------------------------------------------+ +---------------------------------------------+
|  MODULE A: Tactical Index Rebalancer        | |  MODULE B: Strategic Portfolio Stress Test  |
|  - Universe: 16 Large-Cap US Equities       | |  - Synthetic Wholesale Banking Book ($1.8B) |
|  - Recency & Impact Weighted Sentiment      | |  - Multi-Asset: Loans, Bonds, Swaps, FX, Eq |
|  - Dynamic Weight Tilt with 3%-10% Caps     | |  - Automated Trigger Rule (Impact >= 7.0)   |
|  - 1-Year Historical Daily Replay           | |  - Multi-Factor Shock & Valuation Engine    |
|  - Backtest vs Equal-Weight Benchmark       | |  - Pre/Post Stress Value & Loss Breakdown   |
+---------------------------------------------+ +---------------------------------------------+
                          \                                             /
                           \                                           /
                            v                                         v
+---------------------------------------------------------------------------------------------+
|                     INTERACTIVE STREAMLIT DASHBOARD & FASTAPI ENDPOINTS                     |
|            [Tab 1: Index Rebalancer]   [Tab 2: Stress Testing]   [Tab 3: Signal Feed]       |
+---------------------------------------------------------------------------------------------+
```

### Downstream Applications

1. **Module A: Tactical Index Rebalancer**  
   Manages a mock equity index of 16 prominent US equities. Daily portfolio weights are dynamically adjusted: stocks exhibiting positive sentiment are overweight, while stocks with negative sentiment are underweight. Weights respect strict risk concentration bounds ($3\% \le w_i \le 10\%$) and budget normalization ($\sum w_i = 100\%$). Includes full historical backtesting against an equal-weight benchmark with excess return, tracking error, and t-statistic significance.

2. **Module B: Strategic Portfolio Stress Testing**  
   Simulates macroeconomic and geopolitical shocks on a synthetic **$1.8 Billion wholesale banking portfolio** spanning 6 asset classes (corporate loans, corporate bonds, sovereign bonds, equities, interest-rate swaps, and FX forwards). When high-impact events are detected ($\text{Impact} \ge 7.0$), automated stress scenarios apply multi-factor shocks across equity prices, yield curves, credit spreads, FX rates, and loan default probabilities (PD/LGD).

---

## 2. Architecture & Tech Stack

![Architecture Diagram](docs/architecture.png)

- **Language / Runtime:** Python 3.11 / 3.14 (macOS, Linux, Windows)
- **NLP & Transformers:** Hugging Face `transformers`, PyTorch, FinBERT (`ProsusAI/finbert`), NLI Zero-Shot (`cross-encoder/nli-distilroberta-base`)
- **Financial Modeling & Analytics:** pandas, numpy, yfinance
- **API Framework:** FastAPI, Uvicorn, Pydantic
- **Dashboard & Visualization:** Streamlit, Plotly Express / Graph Objects
- **Testing & Quality Assurance:** pytest, httpx

```
vit-sarthak_satish_borekar-hackathon/
├── README.md               # Complete documentation, setup, results & architecture
├── requirements.txt        # Python dependency manifest
├── LICENSE                 # MIT Open-Source License
├── data/
│   ├── sample_news.csv     # Synthetic news headlines for offline execution
│   ├── sample_tweets.csv   # Stratified sample of 560 stock tweets across 16 tickers
│   ├── prices.csv          # 1-year daily adjusted closing prices (2021-2022)
│   ├── portfolio.csv       # Synthetic wholesale banking portfolio (loans, bonds, swaps, FX)
│   ├── event_eval.csv      # Ground-truth evaluation set for event classification
│   └── signals.json        # Machine-readable output generated by the NLP engine
├── docs/
│   ├── architecture.png    # High-resolution architectural diagram
│   └── presentation.pdf    # 7-slide case study presentation deck
├── src/
│   ├── ingestion/          # Unified document schemas, News/Twitter loaders, Ticker regex matcher
│   ├── engine/             # FinBERT sentiment, Zero-shot classifier, Impact formula, Export CLI
│   ├── api/                # FastAPI application with filtering and ticker endpoints
│   ├── rebalancer/         # Universe config, dynamic weight tilt, bounds capping, backtesting
│   ├── stress/             # Portfolio loader, event shock scenarios, multi-asset valuation engine
│   └── dashboard/          # Streamlit 3-tab interactive analytics dashboard
└── tests/                  # Pytest test suite (34 unit & integration tests)
```

---

## 3. Dataset Used

| Dataset / File | Source / Origin | Volume & Nature | Usage & Assumptions |
|---|---|---|---|
| `data/sample_news.csv` | **Synthetic** | 8 curated financial headlines | Serves as deterministic offline news fallback when NewsAPI key is not present. |
| NewsAPI (`newsapi.org`) | **Public API** (Free tier) | Real-time global headlines | Live news feed queried on-demand with `--live` flag. |
| `data/sample_tweets.csv` | **Kaggle** ([Stock Tweets Dataset](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction), CC0 1.0) | 560 stratified tweets (up to 40 per ticker) | Historical social media feed covering 2021-09-30 to 2022-09-30 across the 16 index tickers. |
| `data/prices.csv` | **Yahoo Finance** via `yfinance` | 252 trading days $\times$ 16 tickers | Daily adjusted close prices from 2021-09-30 to 2022-09-29 matching the tweets timeline. |
| `data/portfolio.csv` | **Synthetic Wholesale Banking Book** | 15 multi-asset institutional positions ($1.8B gross notional) | Covers amortized cost loans with PD/LGD, fixed-income bonds with duration/convexity, IR swaps, and FX forwards. |
| `data/event_eval.csv` | **Curated Benchmark** | 20 labeled financial event texts | Ground-truth validation dataset for the zero-shot + keyword event classifier. |
| `data/signals.json` | **Engine Output** | 568 structured records | Pre-generated risk signals ready for instant consumption by downstream modules. |

### Key Assumptions & Modeling Decisions
- **Look-Ahead Bias Prevention:** In Module A backtesting, sentiment signals generated on or before day $t$ are applied strictly to day $t+1$ asset returns.
- **Source Weighting:** News wire text carries higher institutional credibility ($1.0$) than social media posts ($0.7$) in the impact scoring formula.
- **Valuation Approximations:** Bond price moves utilize a 2nd-order modified duration-convexity Taylor approximation. Corporate loans reflect extra Expected Credit Loss ($\Delta \text{ECL} = \text{Notional} \times (\text{PD}_{\text{stressed}} - \text{PD}_{\text{base}}) \times \text{LGD}$). Derivatives are modeled without counterparty credit risk.
- **Zero Proprietary Data:** All data is strictly open-source, synthetic, or publicly available; no real confidential client data is used.

---

## 4. Quickstart & Installation

### Prerequisites
- Python 3.11 or Python 3.14 on macOS / Linux / Windows
- Git

### Step-by-Step Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/dev-sarthak7/vit-sarthak_satish_borekar-hackathon.git
cd vit-sarthak_satish_borekar-hackathon

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Components

```bash
# Step A: Run the AI/NLP Risk Engine (generates data/signals.json)
python -m src.engine.export

# (Optional: Ingest live headlines from NewsAPI if NEWSAPI_KEY is set in .env)
# python -m src.engine.export --live

# Step B: Launch the FastAPI REST Service
uvicorn src.api.main:app --reload --port 8000
# -> Interactive Swagger API docs available at: http://127.0.0.1:8000/docs

# Step C: Launch the Interactive Streamlit Dashboard (in a new terminal)
streamlit run src/dashboard/app.py
# -> Web UI accessible at: http://localhost:8501

# Step D: Run the Full Test Suite
python -m pytest
```

---

## 5. Key Results & Domain Impact

### Prototype Outputs & Demonstrations

1. **AI/NLP Signal Pipeline:**
   - Evaluated against `data/event_eval.csv`, the hybrid zero-shot + keyword classifier achieves high precision across nuanced financial events, routing non-event noise to `Other` or `Market Move`.
   - The REST API provides sub-millisecond filtering on tickers, event types, and impact cutoffs.

2. **Module A — Tactical Rebalancing Backtest:**
   - Over the 2021–2022 test period (a challenging tech bear market), the sentiment-tilted portfolio dynamically reduced exposure to negative-sentiment names while maintaining compliance with the $3\%-10\%$ individual asset bounds.
   - The backtest dashboard computes annualized excess returns, tracking error, and t-statistic significance to provide rigorous transparency rather than naive backtest claims.

3. **Module B — Wholesale Banking Stress Testing:**
   - Ingesting a Geopolitical shock signal ($\text{Impact} = 8.5$) triggers a scenario with $-12.1\%$ equity drop, $-36\text{ bps}$ rate decline, $+97\text{ bps}$ credit spread widening, and $+3.6\%$ USD appreciation.
   - The valuation engine computes a net portfolio loss of $-\$42.8\text{M}$ across the $\$1.8\text{B}$ balance sheet, decomposing losses into mark-to-market bond price drops, derivative hedging offsets, and extra loan default provisions ($\Delta \text{ECL}$).

### Business & Domain Impact

- **Operational Efficiency:** Automates the processing of thousands of daily news articles and social posts into structured risk metrics, saving hours of manual analyst screening.
- **Proactive Risk Management:** Enables quantitative risk officers to instantly identify high-severity events ($\text{Impact} \ge 7$) and stress test multi-asset balance sheets before market close.
- **Governance & Transparency:** Avoids "black box" decisions by exposing explicit sentiment scores, event classification confidences, and deterministic shock formulas.

### Limitations & Next Steps
- **Model Training:** Current impact scoring relies on a structured heuristic formula; future iterations can calibrate impact against historical tick-by-tick abnormal asset returns using proprietary high-frequency data.
- **Streaming Pipeline:** Extend ingestion from batch processing to a real-time Kafka or WebSocket streaming pipeline for sub-second trade signal generation.

---

## 6. AI Usage & Academic Integrity Statement

In accordance with Section 6 of the S&P Global & CRISIL Hackathon guidelines, AI assistance (Claude / Anthropic & Google Antigravity) was utilized as an ideation and coding partner for drafting boilerplate architecture, writing tests, and formatting documentation. All financial modeling logic, valuation formulas, event rules, and code implementations were verified, debugged, tested, and validated by the candidate.