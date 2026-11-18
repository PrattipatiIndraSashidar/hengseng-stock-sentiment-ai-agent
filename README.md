# HengSeng Sentiment Price Prediction Agent

An end-to-end, multi-agent financial sentiment analysis and next-trading-day (`n+1`) closing price prediction platform covering **10 equities** across the **Hong Kong Stock Exchange (HKEX)** and **Korea Exchange (KRX)**.

It combines a **React + TypeScript + Express + Vite** dark-terminal web application (port `3000`) with a modular **8-agent Python NLP/ML pipeline**, **FastAPI** service, **Streamlit** dashboard, and **SQLAlchemy / SQLite** persistence.

---

## Key Capabilities

- **Multi-Source Data Collection**
  - **Social Media (Apify):** Configured for Hong Kong (**LIHKG**, **Telegram**) and South Korea (**Naver Finance**, **Daum Finance**) with actor permission verification, cursor pagination, rate-limit backoff, and content-hash deduplication.
  - **Financial News (Exa Search API):** Collects recent news articles across earnings, guidance, regulatory updates, and analyst coverage with canonical URL deduplication.
  - **Market OHLCV Prices (Yahoo Finance + Naver Backup):** Cookie-authenticated Yahoo Finance session handshake (`query2`/`query1` failover), `HTTP 429` exponential backoff, local caching, and Naver Finance KRX chart backup.
- **3-Tier Multilingual NLP Sentiment Engine (`EN` / `ZH-HK` / `KO`)**
  1. **Tier 1 (Transformer Runtime):** Hugging Face `cardiffnlp/twitter-xlm-roberta-base-sentiment` when `transformers` and `torch` are installed (`--use-hf`).
  2. **Tier 2 (Default Statistical NLP):** `MultilingualFinancialNLPModel` (`multilingual-tfidf-logreg-hybrid-v2`) — L2-regularized Multinomial Logistic Regression over Unicode script tokens, CJK/Hangul 2–3 character n-grams, negation scoping, contrastive-clause weighting, and financial lexicon priors.
  3. **Tier 3 (Lexicon Fallback):** Deterministic `multilingual-financial-lexicon-v1`.
  - Applies engagement weighting `1 + ln(1 + likes + 2*comments + 3*reposts)` and fuses `55%` social sentiment with `45%` news sentiment (or `100%` of whichever live feed is available).
- **Strict 3-Tier Data Provenance & ML Isolation**
  - Classifies feeds transparently as `LIVE DATA`, `HISTORICAL DATA`, or `SIMULATED DEMO DATA`.
  - Simulated demo records (used only for UI demonstration when a platform enforces anti-bot verification such as Cloudflare Turnstile on LIHKG or when credentials are absent) are **strictly excluded** from ML feature engineering and model training.
- **Leakage-Free Timestamp Alignment & Technical Features**
  - Aligns social posts and news strictly to exchange closing cutoffs (**HKEX `16:00 HKT`**, **KRX `15:30 KST`**, plus weekend/holiday calendars).
  - Engineers causal indicators: `RSI-14`, `MACD(12, 26, 9)`, `SMA-5/10/20`, 10-day rolling volatility, Pivot + 20-day swing Support/Resistance, and explicit-only trading position extraction (`Long` / `Short` / `Hold` / `Not mentioned`).
- **Chronological Walk-Forward Validation & Conformal Intervals**
  - Evaluates **Linear Regression (Ridge)**, **Random Forest Regressor**, and **XGBoost Regressor** predicting stationary next-day return $r_{t+1}$, benchmarked against a **Naive Previous-Close Baseline** ($r = 0$) with split-conformal `80%` and `95%` prediction intervals.
- **Interactive Dashboard & Automated Exports**
  - 7-view React dark terminal (`Overview`, `Hong Kong Market`, `South Korea Market`, `Sentiment Analysis`, `Stock Intelligence`, `Price Prediction`, `Reports`), CSV & PDF report exports, and a 4-page **Project Overview PDF** (`about_hengseng_project.pdf`).

---

## Monitored Equity Universe (10 Stocks)

Configured in `stock_sentiment_agent/config/tickers.json` and editable from the UI **Reports -> Configure Tickers** panel:

| Exchange | Ticker | Company | Currency | Social Sources | News Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HKEX** | `0700.HK` | Tencent | `HKD` | LIHKG, Telegram | Exa Search API |
| **HKEX** | `9988.HK` | Alibaba | `HKD` | LIHKG, Telegram | Exa Search API |
| **HKEX** | `3690.HK` | Meituan | `HKD` | LIHKG, Telegram | Exa Search API |
| **HKEX** | `1810.HK` | Xiaomi | `HKD` | LIHKG, Telegram | Exa Search API |
| **HKEX** | `1299.HK` | AIA Group | `HKD` | LIHKG, Telegram | Exa Search API |
| **KRX** | `005930.KS` | Samsung Electronics | `KRW` | Naver Finance, Daum Finance | Exa Search API |
| **KRX** | `000660.KS` | SK Hynix | `KRW` | Naver Finance, Daum Finance | Exa Search API |
| **KRX** | `035420.KS` | NAVER | `KRW` | Naver Finance, Daum Finance | Exa Search API |
| **KRX** | `005380.KS` | Hyundai Motor | `KRW` | Naver Finance, Daum Finance | Exa Search API |
| **KRX** | `035720.KS` | Kakao | `KRW` | Naver Finance, Daum Finance | Exa Search API |

---

## System Architecture

```text
  [ React 19 + Tailwind + Vite Terminal ]  <--->  [ Express Server (server.ts :3000) ]
                                                              |
                                         +--------------------+--------------------+
                                         |                                         |
                                         v                                         v
                         [ Python PipelineOrchestrator ]                 [ SQLite / SQLAlchemy ]
                                         |                                   (9 SQL Tables)
        +--------------------------------+--------------------------------+
        |                |               |                |               |
        v                v               v                v               v
  1. SocialAgent   2. NewsAgent   3. SentimentAgent  4. IntelAgent   5. FeatureAgent
  (Apify/Direct)     (Exa API)    (EN/ZH-HK/KO NLP)  (20 KPIs, S/R)  (HKT/KST Cutoffs)
                                                                          |
                                                                          v
                         8. ReportingAgent  <--  7. PredictionAgent  <--  6. ValidationAgent
                         (CSV, PDF, Audits)      (n+1 + Conformal CI)     (Walk-Forward vs Naive)
```

### Database Schema (9 Tables)
Defined in `stock_sentiment_agent/database/schema.py`:
`stocks` · `social_posts` · `news_articles` · `stock_prices` · `sentiment_results` · `financial_features` · `predictions` · `model_metrics` · `agent_logs`

---

## Project Structure

```text
.
├── server.ts                          # Express HTTP/WS server + Vite middleware + /api/* bridge
├── vite.config.ts                     # ESM Vite 8 config (import.meta.dirname, proxy WS settings)
├── index.html                         # Frontend HTML entry point
├── package.json                       # Node scripts & frontend/server dependencies
├── .env.example                       # Environment variable template (no secrets)
├── scripts/
│   └── generate_about_pdf.mjs         # Generates 4-page About This Project PDF via jsPDF
├── src/
│   ├── App.tsx                        # Root terminal layout, navigation & pipeline state
│   ├── types.ts                       # TypeScript data models
│   ├── components/
│   │   ├── DashboardPages.tsx         # 7 dashboard views + ticker config & export actions
│   │   └── TerminalCharts.tsx         # SVG/HTML financial, sentiment, error & feature charts
│   └── utils/
│       └── exportReports.ts           # Client-side CSV & jsPDF table exports
└── stock_sentiment_agent/
    ├── main.py                        # Python CLI entrypoint (full or --ticker single run)
    ├── requirements.txt               # Python backend/ML/dashboard dependencies
    ├── config/
    │   ├── settings.py                # Env loader, SQLite fallback, configuration validation
    │   └── tickers.json               # Configurable HKEX & KRX equity universe
    ├── collectors/                    # Apify, LIHKG, Telegram, Naver, Daum, Exa & Price collectors
    ├── preprocessing/                 # Text cleaning, HKT/KST session alignment, technical features
    ├── models/                        # Multilingual NLP, Ridge, RF, XGBoost, Walk-Forward evaluator
    ├── agents/                        # 8 modular agents + PipelineOrchestrator
    ├── database/                      # SQLAlchemy ORM models & SQLite/PostgreSQL manager
    ├── api/server.py                  # Standalone FastAPI server
    ├── dashboard/app.py               # Standalone 7-page Streamlit + Plotly dashboard
    ├── reports/
    │   ├── about_hengseng_project.pdf # 4-page technical project overview PDF
    │   ├── final_assessment_report.md # Integration & validation assessment report
    │   ├── hengseng_report.csv        # Latest exported equity metrics CSV
    │   └── hengseng_report.pdf        # Latest multi-page forecast PDF
    └── tests/
        └── test_pipeline.py           # Python unit & integration tests
```

---

## Environment Configuration

Copy `.env.example` to `.env` and fill in your credentials as needed:

```env
GEMINI_API_KEY="MY_GEMINI_API_KEY"
APP_URL="http://localhost:3000"

# Data Providers (optional — missing keys fall back safely to disclosed demo mode for social/news)
APIFY_API_TOKEN=
EXA_API_KEY=

# Persistence & Model Artifacts
DATABASE_URL="sqlite:///./stock_sentiment_agent/data/hengseng.db"
MODEL_PATH="./stock_sentiment_agent/models/artifacts"
```

- **Security:** API tokens are read strictly server-side and never exposed to the browser UI.
- **SQLite Fallback:** If `DATABASE_URL` is omitted or unreachable, the pipeline automatically uses local SQLite at `stock_sentiment_agent/data/hengseng.db`.

---

## Getting Started

### 1. Run the Full-Stack Terminal (Express + React + Vite on Port 3000)

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

### 2. Run the Python Multi-Agent Pipeline (CLI)

```bash
# Install Python dependencies (recommended in a virtual environment)
pip install -r stock_sentiment_agent/requirements.txt

# Run all 10 tickers
python3 -m stock_sentiment_agent.main

# Run a single ticker
python3 -m stock_sentiment_agent.main --ticker 0700.HK
```

### 3. Optional Standalone Python Interfaces

```bash
# FastAPI server (port 8000)
uvicorn stock_sentiment_agent.api.server:app --reload --port 8000

# Streamlit 7-page dashboard
streamlit run stock_sentiment_agent/dashboard/app.py
```

### 4. Generate the "About This Project" PDF

```bash
node scripts/generate_about_pdf.mjs
```

Outputs `stock_sentiment_agent/reports/about_hengseng_project.pdf` (also downloadable from the **Reports** tab via **Project Overview PDF** or `GET /api/reports/about-pdf`).

---

## Express API Endpoints (`server.ts`)

| Method & Path | Description |
| :--- | :--- |
| `GET /api/health` | Safe environment status (`apifyConfigured`, `exaConfigured`, DB/snapshot status; no secrets) |
| `GET /api/dashboard` | Returns latest pipeline snapshot merged with SQLite `agent_logs` & `predictions` history |
| `POST /api/pipeline/run` | Executes the Python multi-agent pipeline (all tickers or `{ "ticker": "0700.HK" }`) |
| `GET /api/stocks/config` | Returns configured equities from `stock_sentiment_agent/config/tickers.json` |
| `POST /api/stocks/config` | Updates the active equity universe |
| `GET /api/reports/csv` | Downloads `hengseng_report.csv` |
| `GET /api/reports/pdf` | Downloads `hengseng_report.pdf` |
| `GET /api/reports/about-pdf` | Downloads `HengSeng_About_This_Project.pdf` |

---

## Verification & Checks

```bash
# TypeScript type check
npm run lint

# Production frontend build
npm run build

# Generate labeled sample dataset (if running offline unit tests for demo mode)
python3 stock_sentiment_agent/data/build_sample_dataset.py

# Run Python pipeline test suite
python3 -m unittest discover -s stock_sentiment_agent/tests -v
```

---

## Known Limitations

- **Anti-Bot Protections:** Platforms enforcing Cloudflare Turnstile or strict anti-bot controls (such as LIHKG) are never bypassed; unavailable platforms are explicitly reported and excluded from ML training.
- **Transformer Runtime:** When `transformers` and `torch` are not installed, sentiment runs the Tier-2 TF-IDF + Logistic Regression hybrid NLP model with deterministic lexicon fallback.
- **Short-Horizon Market Noise:** Daily large-cap equity returns carry high stochastic variance; regularized models and conformal uncertainty bands are used to prevent overconfident point forecasts.
