# HengSeng Sentiment Price Prediction Agent

An AI-powered multilingual financial sentiment analysis and next-day (`n+1`) closing price prediction platform covering **10 equities** across the **Hong Kong Stock Exchange (HKEX)** and **South Korea Exchange (KRX)**.

---

## 1. Project Overview & Problem Statement

Retail and institutional discussion across Hong Kong (LIHKG, Telegram) and South Korea (Naver Finance, Daum Finance) contains timely multilingual market signals in Traditional Chinese, Korean, and English, while financial news sources capture earnings, regulatory updates, and analyst revisions. **HengSeng Sentiment Price Prediction Agent** unifies:

1. **Multi-source social collection** via Apify actors (2 sources per exchange).
2. **Financial news retrieval** via the official Exa Search API.
3. **Multilingual engagement-weighted NLP** (`XLM-RoBERTa` with deterministic multilingual financial lexicon fallback).
4. **Leakage-free exchange timestamp alignment** (HKT `16:00` / KST `15:30` close cutoffs and holiday calendars).
5. **Chronological walk-forward ML forecasting** comparing Linear Regression, Random Forest, XGBoost, and a Naive Previous-Close Baseline.
6. **Interactive dark-terminal dashboard** (both a Streamlit + Plotly app and a live React + Express terminal) with CSV & PDF exports.

---

## 2. Monitored Equity Universe (Configurable)

Configured in `stock_sentiment_agent/config/tickers.json` and `config/settings.py`:

| Exchange | Ticker | Company | Social Sources |
| --- | --- | --- | --- |
| HKEX | `0700.HK` | Tencent | LIHKG, Telegram |
| HKEX | `9988.HK` | Alibaba | LIHKG, Telegram |
| HKEX | `3690.HK` | Meituan | LIHKG, Telegram |
| HKEX | `1810.HK` | Xiaomi | LIHKG, Telegram |
| HKEX | `1299.HK` | AIA Group | LIHKG, Telegram |
| KRX | `005930.KS` | Samsung Electronics | Naver Finance, Daum Finance |
| KRX | `000660.KS` | SK Hynix | Naver Finance, Daum Finance |
| KRX | `035420.KS` | NAVER | Naver Finance, Daum Finance |
| KRX | `005380.KS` | Hyundai Motor | Naver Finance, Daum Finance |
| KRX | `035720.KS` | Kakao | Naver Finance, Daum Finance |

---

## 3. Multi-Agent Architecture

Eight independent agents coordinated by `PipelineOrchestrator`:

1. **Social Data Collection Agent** (`agents/social_agent.py`) — Paginated Apify scraping, raw JSON archival, deduplication, and explicit simulated-mode disclosure.
2. **Financial News Research Agent** (`agents/news_agent.py`) — Exa Search API queries across earnings, revenue, regulatory, and analyst keywords with URL deduplication.
3. **Sentiment Analysis Agent** (`agents/sentiment_agent.py`) — Multilingual scoring (EN/ZH/KO), engagement weighting (`1 + log1p(likes + 2*comments + 3*reposts)`), and transparent 55% social / 45% news fusion.
4. **Financial Intelligence Agent** (`agents/financial_agent.py`) — Extracts all 20 intelligence fields including explicit-only trading positions (`Long`/`Short`/`Hold`/`Not mentioned`) and Pivot/Swing Support & Resistance.
5. **Feature Engineering Agent** (`agents/financial_agent.py` + `preprocessing/feature_engineering.py`) — Computes RSI-14, MACD, SMA-5/10/20, 10d volatility, mention growth, and news recency aligned strictly to session cutoffs.
6. **Model Validation Agent** (`agents/validation_agent.py`) — Expanding-window walk-forward out-of-sample evaluation with MAE, RMSE, MAPE, Directional Accuracy, and realized reconciliation.
7. **Price Prediction Agent** (`agents/prediction_agent.py`) — Trains the best validated model on causal history to forecast `Close[n+1]` and saves Joblib artifacts.
8. **Report Generation Agent** (`agents/reporting_agent.py`) — Produces exchange-wise rankings across 5 criteria plus CSV and PDF exports.

---

## 4. Platform Compliance & Scraping Limitations

- **LIHKG & Daum/Naver Forums**: Automated unauthenticated scraping may encounter Cloudflare protections, SPA rendering limits, or site Terms of Service restrictions. When live Apify actors are blocked or `APIFY_API_TOKEN` / `EXA_API_KEY` are absent, the collectors never fabricate live scraping success—they log the restriction and explicitly load labeled demonstration records (`is_simulated=True`).
- Missing author metrics (e.g., LIHKG/Naver boards without follower counts) are preserved as `NULL`/`None` rather than fabricated.

---

## 5. Database Design

Nine relational SQL tables defined in `database/schema.py` and `database/db.py` (supporting PostgreSQL or SQLite):

`stocks` · `social_posts` · `news_articles` · `stock_prices` · `sentiment_results` · `financial_features` · `predictions` · `model_metrics` · `agent_logs`

---

## 6. Installation & Configuration

```bash
cd stock_sentiment_agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env`:
```env
APIFY_API_TOKEN=your_apify_token
EXA_API_KEY=your_exa_api_key
DATABASE_URL=sqlite:///./stock_sentiment_agent/data/hengseng.db
MODEL_PATH=./stock_sentiment_agent/models/artifacts
```

---

## 7. Running the Project, Training & Validation

### Run Full Multi-Agent Pipeline (CLI)
```bash
python -m stock_sentiment_agent.main
```

### Run FastAPI Server
```bash
uvicorn stock_sentiment_agent.api.server:app --reload --port 8000
```

### Run Streamlit 7-Page Dashboard
```bash
streamlit run stock_sentiment_agent/dashboard/app.py
```

### Run Test Suite
```bash
python -m unittest discover -s stock_sentiment_agent/tests -v
```

---

## 8. Limitations & Future Improvements

- **Sample / Demo Mode**: Without `APIFY_API_TOKEN` and `EXA_API_KEY`, social and news feeds run in clearly disclosed Demonstration Mode.
- **Short-Horizon Features**: Walk-forward validation accuracy improves as more live daily snapshots accumulate in the SQL database.
- **Future Work**: Intraday 15-minute order-book alignment, SEC/HKEX/DART official filing parsers, and transformer fine-tuning on labeled Cantonese/Korean financial threads.
