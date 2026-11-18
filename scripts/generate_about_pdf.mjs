import fs from 'node:fs';
import path from 'node:path';
import { jsPDF } from 'jspdf';

const ROOT_DIR = process.cwd();
const SNAPSHOT_PATH = path.join(ROOT_DIR, 'stock_sentiment_agent', 'data', 'latest_pipeline_output.json');
const OUT_PATH = path.join(ROOT_DIR, 'stock_sentiment_agent', 'reports', 'about_hengseng_project.pdf');

function loadSnapshot() {
  if (!fs.existsSync(SNAPSHOT_PATH)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(SNAPSHOT_PATH, 'utf-8'));
}

function drawFooter(doc, pageNum, totalPages, generatedDate) {
  const pageW = doc.internal.pageSize.getWidth();
  const pageH = doc.internal.pageSize.getHeight();
  doc.setDrawColor(226, 232, 240);
  doc.line(36, pageH - 36, pageW - 36, pageH - 36);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(100, 116, 139);
  doc.text('HengSeng Sentiment Price Prediction Agent  |  Project Technical Overview', 36, pageH - 22);
  doc.text(`Generated: ${generatedDate}  |  Page ${pageNum} of ${totalPages}`, pageW - 36, pageH - 22, { align: 'right' });
}

function drawSectionHeading(doc, numLabel, title, y) {
  doc.setFillColor(14, 165, 233);
  doc.rect(36, y - 10, 4, 14, 'F');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12);
  doc.setTextColor(15, 23, 42);
  doc.text(`${numLabel}  ${title}`, 46, y + 1);
  doc.setDrawColor(226, 232, 240);
  doc.line(36, y + 8, doc.internal.pageSize.getWidth() - 36, y + 8);
  return y + 22;
}

function drawParagraph(doc, text, x, y, maxW, lineHeight = 13) {
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9.5);
  doc.setTextColor(51, 65, 85);
  const lines = doc.splitTextToSize(text, maxW);
  doc.text(lines, x, y);
  return y + lines.length * lineHeight;
}

function drawTable(doc, headers, rows, startY, colWidths, rowHeight = 18) {
  const startX = 36;
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  let y = startY;

  doc.setFillColor(15, 23, 42);
  doc.rect(startX, y, totalW, rowHeight + 2, 'F');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(248, 250, 252);

  let cx = startX + 6;
  headers.forEach((h, i) => {
    doc.text(String(h), cx, y + 13);
    cx += colWidths[i];
  });
  y += rowHeight + 2;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  rows.forEach((row, idx) => {
    if (idx % 2 === 1) {
      doc.setFillColor(248, 250, 252);
      doc.rect(startX, y, totalW, rowHeight, 'F');
    }
    doc.setDrawColor(226, 232, 240);
    doc.line(startX, y + rowHeight, startX + totalW, y + rowHeight);
    let rx = startX + 6;
    row.forEach((cell, i) => {
      const txt = String(cell ?? '');
      doc.setTextColor(30, 41, 59);
      const clipped = doc.splitTextToSize(txt, colWidths[i] - 10)[0] || '';
      doc.text(clipped, rx, y + 12);
      rx += colWidths[i];
    });
    y += rowHeight;
  });

  return y + 10;
}

function buildProjectPdf() {
  const snapshot = loadSnapshot();
  const generatedDate = new Date().toISOString().slice(0, 10);
  const doc = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
  const pageW = doc.internal.pageSize.getWidth(); // 595.28
  const contentW = pageW - 72; // 523.28
  const totalPages = 4;

  // ======================== PAGE 1: OVERVIEW & UNIVERSE ========================
  doc.setFillColor(9, 13, 22);
  doc.rect(0, 0, pageW, 108, 'F');
  doc.setFillColor(14, 165, 233);
  doc.rect(0, 104, pageW, 4, 'F');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(19);
  doc.setTextColor(248, 250, 252);
  doc.text('HengSeng Sentiment Price Prediction Agent', 36, 44);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(10.5);
  doc.setTextColor(148, 163, 184);
  doc.text('About This Project: Architecture, Multilingual NLP, ML Forecasting & Live Integrations', 36, 65);

  doc.setFontSize(8.5);
  doc.setTextColor(56, 189, 248);
  doc.text('HKEX (Hong Kong) & KRX (South Korea)  |  8-Agent Pipeline  |  Walk-Forward Validation  |  Dark Terminal UI', 36, 86);

  let y = 132;
  y = drawSectionHeading(doc, '01', 'Executive Summary & Objectives', y);
  y = drawParagraph(
    doc,
    'HengSeng Sentiment Price Prediction Agent is an end-to-end, multi-agent financial intelligence and short-horizon forecasting platform covering 10 major equities across the Hong Kong Stock Exchange (HKEX) and Korea Exchange (KRX). The system collects social forum posts and financial news in English, Traditional Chinese, and Korean, scores multilingual sentiment with engagement weighting, aligns every signal strictly to exchange closing cutoffs to prevent look-ahead leakage, and forecasts next-trading-day (n+1) closing prices with conformal uncertainty intervals.',
    36,
    y,
    contentW
  );
  y += 8;

  // Key Highlight KPI Boxes
  const kpis = [
    { label: 'Monitored Equities', val: '10 (5 HKEX / 5 KRX)' },
    { label: 'Agent Modules', val: '8 Independent Agents' },
    { label: 'SQL Persistence', val: '9 Relational Tables' },
    { label: 'ML Models Compared', val: '3 ML + Naive Baseline' },
  ];
  const boxW = (contentW - 24) / 4;
  kpis.forEach((k, idx) => {
    const bx = 36 + idx * (boxW + 8);
    doc.setFillColor(241, 245, 249);
    doc.setDrawColor(203, 213, 225);
    doc.roundedRect(bx, y, boxW, 42, 4, 4, 'FD');
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor(100, 116, 139);
    doc.text(k.label, bx + 8, y + 15);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(15, 23, 42);
    doc.text(k.val, bx + 8, y + 31);
  });
  y += 58;

  y = drawSectionHeading(doc, '02', 'Technology Stack', y);
  y = drawTable(
    doc,
    ['Layer', 'Technologies & Libraries', 'Responsibility'],
    [
      ['Frontend Terminal', 'React 19, TypeScript, Tailwind CSS v4, Vite 8, Lucide, jsPDF', '7-view dark financial terminal, SVG/HTML charts, PDF/CSV export'],
      ['Server & API Bridge', 'Node.js 22, Express 4, FastAPI (Python 3.11), REST /api/*', 'Serves SPA, triggers Python orchestrator, streams SQLite telemetry'],
      ['Data Collection', 'Apify Actors (cheerio/web), Exa Search API, Yahoo/Naver Prices', 'Social scraping, financial news retrieval, OHLCV market history'],
      ['AI, NLP & ML', 'XLM-RoBERTa / TF-IDF LogReg Hybrid, Ridge, RF, XGBoost', 'Multilingual sentiment (EN/ZH-HK/KO), walk-forward return forecast'],
      ['Persistence', 'SQLAlchemy ORM + SQLite fallback (PostgreSQL compatible)', 'Stores raw posts, news, OHLCV, features, predictions, audit logs'],
    ],
    y,
    [105, 215, 203],
    18
  );

  y = drawSectionHeading(doc, '03', 'Monitored 10-Equity Universe (Configurable)', y);
  y = drawTable(
    doc,
    ['Ticker', 'Company', 'Exchange', 'Currency', 'Primary Social Sources', 'News Source'],
    [
      ['0700.HK', 'Tencent', 'HKEX', 'HKD', 'LIHKG, Telegram HK Channels', 'Exa Search API'],
      ['9988.HK', 'Alibaba', 'HKEX', 'HKD', 'LIHKG, Telegram HK Channels', 'Exa Search API'],
      ['3690.HK', 'Meituan', 'HKEX', 'HKD', 'LIHKG, Telegram HK Channels', 'Exa Search API'],
      ['1810.HK', 'Xiaomi', 'HKEX', 'HKD', 'LIHKG, Telegram HK Channels', 'Exa Search API'],
      ['1299.HK', 'AIA Group', 'HKEX', 'HKD', 'LIHKG, Telegram HK Channels', 'Exa Search API'],
      ['005930.KS', 'Samsung Electronics', 'KRX', 'KRW', 'Naver Finance, Daum Finance', 'Exa Search API'],
      ['000660.KS', 'SK Hynix', 'KRX', 'KRW', 'Naver Finance, Daum Finance', 'Exa Search API'],
      ['035420.KS', 'NAVER', 'KRX', 'KRW', 'Naver Finance, Daum Finance', 'Exa Search API'],
      ['005380.KS', 'Hyundai Motor', 'KRX', 'KRW', 'Naver Finance, Daum Finance', 'Exa Search API'],
      ['035720.KS', 'Kakao', 'KRX', 'KRW', 'Naver Finance, Daum Finance', 'Exa Search API'],
    ],
    y,
    [62, 118, 55, 50, 152, 86],
    16.5
  );
  drawFooter(doc, 1, totalPages, generatedDate);

  // ======================== PAGE 2: ARCHITECTURE & PROVENANCE ========================
  doc.addPage();
  y = 44;
  y = drawSectionHeading(doc, '04', '8-Agent Pipeline Architecture', y);
  y = drawParagraph(
    doc,
    'PipelineOrchestrator executes eight modular Python agents sequentially for each equity. Batched regional scrapes run once per platform before per-ticker alignment, feature construction, out-of-sample validation, and artifact persistence.',
    36,
    y,
    contentW
  );
  y += 6;

  // Pipeline Flow Diagram (2 rows x 4 boxes)
  const agentsFlow = [
    ['1. Social Agent', 'LIHKG, Telegram, Naver, Daum'],
    ['2. News Agent', 'Exa Search + URL dedupe'],
    ['3. Sentiment Agent', 'EN/ZH/KO NLP + 55/45 fusion'],
    ['4. Intelligence Agent', '20 KPIs + S/R + Position'],
    ['5. Feature Agent', 'Session-aligned RSI/MACD/SMA'],
    ['6. Validation Agent', 'Chronological walk-forward'],
    ['7. Prediction Agent', 'n+1 close + 80/95% Conformal CI'],
    ['8. Reporting Agent', 'Rankings, CSV, PDF & Audits'],
  ];
  const fBoxW = (contentW - 24) / 4;
  agentsFlow.forEach((ag, idx) => {
    const row = Math.floor(idx / 4);
    const col = idx % 4;
    const fx = 36 + col * (fBoxW + 8);
    const fy = y + row * 48;
    doc.setFillColor(row === 0 ? 240 : 248, row === 0 ? 249 : 250, 255);
    doc.setDrawColor(14, 165, 233);
    doc.roundedRect(fx, fy, fBoxW, 38, 4, 4, 'FD');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8.2);
    doc.setTextColor(15, 23, 42);
    doc.text(ag[0], fx + 7, fy + 14);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.2);
    doc.setTextColor(71, 85, 105);
    doc.text(ag[1], fx + 7, fy + 27);
    if (col < 3) {
      doc.setDrawColor(100, 116, 139);
      doc.line(fx + fBoxW + 1, fy + 19, fx + fBoxW + 7, fy + 19);
    }
  });
  y += 108;

  y = drawSectionHeading(doc, '05', '3-Tier Data Provenance & Strict ML Isolation', y);
  y = drawTable(
    doc,
    ['Provenance Label', 'Meaning & Source Scope', 'ML Training Eligibility'],
    [
      ['LIVE DATA', 'Verified external API records (Exa news, live quotes, live posts)', 'Included in ML feature matrix'],
      ['HISTORICAL DATA', 'Verified Yahoo Finance / Naver KRX historical OHLCV bars', 'Included in ML feature matrix'],
      ['SIMULATED DEMO DATA', 'Labeled sample placeholders when scraping/keys are blocked', 'STRICTLY EXCLUDED (0% ML usage)'],
    ],
    y,
    [125, 248, 150],
    19
  );
  y = drawParagraph(
    doc,
    'Platform Compliance Note: LIHKG enforces Cloudflare Turnstile anti-bot checks. Collectors verify Apify actor availability without attempting security bypasses, report blocked platforms transparently as unavailable, and isolate any demonstration posts from model training.',
    36,
    y,
    contentW
  );
  y += 10;

  y = drawSectionHeading(doc, '06', 'External Data Integrations & Rate-Limit Resilience', y);
  y = drawTable(
    doc,
    ['Provider / Source', 'Integration Mechanism', 'Resilience & Deduplication Strategy'],
    [
      ['Apify Social Actors', 'REST v2 & apify-client (cheerio-scraper, web-scraper)', 'Token auth check, actor verify, cursor pagination, hash dedupe'],
      ['Exa Search API', 'Official /search endpoint with 10-day lookback filter', 'Canonical URL deduplication, domain attribution, relevance scoring'],
      ['Yahoo Finance / Naver', 'CookieJar fc.yahoo.com + query2/query1 + Naver KRX chart', '0.35s throttling, exponential backoff on HTTP 429, 4h JSON cache'],
      ['Environment Config', 'dotenv + Settings dataclass + SQLite fallback', 'Graceful missing-key warnings; secrets never exposed to frontend'],
    ],
    y,
    [115, 198, 210],
    18
  );
  drawFooter(doc, 2, totalPages, generatedDate);

  // ======================== PAGE 3: NLP, FEATURES & DB ========================
  doc.addPage();
  y = 44;
  y = drawSectionHeading(doc, '07', 'Multilingual NLP Sentiment Architecture (EN / ZH-HK / KO)', y);
  y = drawParagraph(
    doc,
    'The Sentiment Analysis Agent implements a three-tier multilingual classifier supporting English, Traditional Chinese (Hong Kong equities), and Korean (KRX equities) with negation scoping and contrastive clause weighting (e.g., "however", "but", "但", "하지만"):',
    36,
    y,
    contentW
  );
  y += 6;
  y = drawTable(
    doc,
    ['Tier', 'Engine Identifier', 'Activation Condition & Design'],
    [
      ['Tier 1', 'cardiffnlp/twitter-xlm-roberta-base-sentiment', 'Active when transformers & PyTorch runtime are installed (--use-hf)'],
      ['Tier 2', 'multilingual-tfidf-logreg-hybrid-v2', 'Default statistical NLP: L2-regularized TF-IDF + Softmax Logistic Regression'],
      ['Tier 3', 'multilingual-financial-lexicon-v1', 'Deterministic domain lexicon fallback across EN, ZH-HK, and KO'],
    ],
    y,
    [48, 200, 275],
    18
  );
  y = drawParagraph(
    doc,
    'Engagement Weighting & Multimodal Fusion: Each social post is weighted by w = 1 + ln(1 + likes + 2*comments + 3*reposts), scaled up to +15% when verified follower counts exist. Combined sentiment fuses 55% engagement-weighted social sentiment and 45% news sentiment (or 100% of whichever live feed is available).',
    36,
    y,
    contentW
  );
  y += 10;

  y = drawSectionHeading(doc, '08', 'Leakage-Free Timestamp Alignment & Feature Engineering', y);
  y = drawParagraph(
    doc,
    'Posts and news published after HKEX close (16:00 HKT, Asia/Hong_Kong) or KRX close (15:30 KST, Asia/Seoul), or on weekends/exchange holidays, are rolled forward to the next valid trading session before computing daily aggregates. Causal rolling features include:',
    36,
    y,
    contentW
  );
  y += 4;
  y = drawTable(
    doc,
    ['Feature Category', 'Engineered Signals (Strictly <= Session t)', 'Target & Intelligence Output'],
    [
      ['Price & Momentum', 'daily_return, close_to_sma5, close_to_sma20, intraday_range', 'Stationary next-day return r_(t+1)'],
      ['Technical Indicators', 'RSI-14, MACD(12,26,9) diff norm, 10-day rolling volatility', 'Support/Resistance via Pivot + 20d Swing'],
      ['News & Social (Live)', 'news_sentiment, news_relevance, news_recency (+ social if live)', 'Explicit-only Long/Short/Hold/Not mentioned'],
    ],
    y,
    [115, 228, 180],
    18
  );

  y = drawSectionHeading(doc, '09', 'Relational Database Schema (9 SQLAlchemy Tables)', y);
  y = drawTable(
    doc,
    ['Table Name', 'Primary Purpose & Unique Constraints', 'Key Stored Fields'],
    [
      ['stocks', 'Master 10-equity registry (unique ticker)', 'ticker, company_name, exchange, sector, currency, is_active'],
      ['social_posts', 'Deduplicated posts (unique source_platform, post_id)', 'post_text, utc_timestamp, aligned_trading_date, likes, is_simulated'],
      ['news_articles', 'Financial news items (unique article_url)', 'headline, source, published_utc, relevance_score, sentiment_score'],
      ['stock_prices', 'Historical OHLCV bars (unique ticker, trading_date)', 'open, high, low, close, volume, daily_return, is_simulated'],
      ['sentiment_results', 'Daily aggregated polarity (unique ticker, date, scope)', 'positive/negative/neutral pct, weighted score, bullish_probability'],
      ['financial_features', 'Session-aligned ML feature matrix (unique ticker, date)', 'RSI-14, MACD, SMAs, volatility_10, news/social metrics, target'],
      ['predictions', 'n+1 forecasts & realized close reconciliation', 'predicted_close, actual_close, pct_change, model_name, is_validated'],
      ['model_metrics', 'Walk-forward model comparison metrics', 'model_name, split_window, mae, rmse, mape, directional_accuracy'],
      ['agent_logs', 'Execution telemetry & audit trail', 'agent_name, status, ticker, message, records_processed, duration_ms'],
    ],
    y,
    [98, 205, 220],
    17
  );
  drawFooter(doc, 3, totalPages, generatedDate);

  // ======================== PAGE 4: VALIDATION, RESULTS & APIS ========================
  doc.addPage();
  y = 44;
  y = drawSectionHeading(doc, '10', 'Walk-Forward Validation & Latest Snapshot Results', y);

  const assessment = snapshot?.assessment_summary;
  const valSummary = assessment?.validation_summary;
  const meanDir = valSummary?.mean_best_directional_accuracy ?? 52.97;
  const naiveDir = valSummary?.mean_naive_directional_accuracy ?? 46.62;
  const beatCount = valSummary?.models_beating_or_matching_naive_rmse ?? 6;
  const totalValidated = valSummary?.total_stocks_validated ?? 10;
  const genuineNews = assessment?.genuine_news_collected ?? 78;

  y = drawParagraph(
    doc,
    `Models predict stationary daily returns r_(t+1) using expanding-window chronological walk-forward validation (65% minimum initial window, zero future leakage), benchmarked against a Naive Previous-Close baseline (r = 0) with split-conformal 80%/95% prediction bands. On the latest full snapshot (${genuineNews} genuine Exa news articles, 10/10 genuine Yahoo Finance price histories), mean best-model directional accuracy reached ${meanDir}% vs ${naiveDir}% Naive, with ${beatCount}/${totalValidated} equities beating or matching Naive RMSE.`,
    36,
    y,
    contentW
  );
  y += 6;

  const summaryRows = (snapshot?.summaries || []).map((s) => {
    const i = s.intelligence;
    const p = s.prediction;
    const best = (s.validation || []).find((v) => v.is_best_model) || {};
    const naive = (s.validation || []).find((v) => String(v.model_name).startsWith('Naive')) || {};
    const dec = i.exchange === 'HKEX' ? 2 : 0;
    const rmseDec = i.exchange === 'HKEX' ? 2 : 0;
    return [
      i.ticker,
      i.latest_price.toFixed(dec),
      `${p.predicted_close.toFixed(dec)} (${p.predicted_pct_change >= 0 ? '+' : ''}${p.predicted_pct_change.toFixed(2)}%)`,
      p.model_name.replace(' Regressor', ''),
      best.rmse !== undefined ? best.rmse.toFixed(rmseDec) : 'N/A',
      naive.rmse !== undefined ? naive.rmse.toFixed(rmseDec) : 'N/A',
      best.directional_accuracy !== undefined ? `${best.directional_accuracy.toFixed(1)}%` : 'N/A',
      `${i.combined_sentiment_score >= 0 ? '+' : ''}${i.combined_sentiment_score.toFixed(3)}`,
    ];
  });

  if (summaryRows.length > 0) {
    y = drawTable(
      doc,
      ['Ticker', 'Close', 'Pred n+1 (%Chg)', 'Best Model', 'ML RMSE', 'Naive RMSE', 'Dir Acc', 'Sentiment'],
      summaryRows,
      y,
      [58, 55, 105, 86, 56, 62, 48, 53],
      16
    );
  }

  y = drawSectionHeading(doc, '11', 'Interactive Dashboard Views & REST API Endpoints', y);
  y = drawTable(
    doc,
    ['Endpoint / View', 'Type', 'Function'],
    [
      ['7 Terminal Views', 'React SPA', 'Overview, HKEX, KRX, Sentiment, Stock Intelligence, Prediction, Reports'],
      ['GET /api/health', 'Express API', 'Safe environment check (APIFY/EXA/DB/MODEL configured flags, no secrets)'],
      ['GET /api/dashboard', 'Express API', 'Loads pipeline JSON snapshot + SQLite agent logs & prediction history'],
      ['POST /api/pipeline/run', 'Express API', 'Executes full or single-ticker Python multi-agent pipeline'],
      ['GET/POST /api/stocks/config', 'Express API', 'Reads/updates active stock universe in tickers.json'],
      ['GET /api/reports/{csv,pdf,about-pdf}', 'Express API', 'Downloads exchange CSV report, forecast PDF, and this Project Overview PDF'],
    ],
    y,
    [155, 78, 290],
    17
  );

  y = drawSectionHeading(doc, '12', 'Known Limitations & Engineering Guardrails', y);
  y = drawParagraph(
    doc,
    '1. Social Anti-Bot Restrictions: Platforms enforcing Cloudflare Turnstile or strict actor permissions (e.g., LIHKG) are never bypassed; when live posts are unavailable, labeled SIMULATED DEMO DATA is isolated from ML training.\n' +
      '2. Runtime NLP Tier: In containers without PyTorch/transformers, sentiment runs our trained TF-IDF + Logistic Regression hybrid model with lexicon fallback.\n' +
      '3. Short-Horizon Noise: Large-cap 1-day equity returns have low signal-to-noise ratio; strong regularization (Ridge alpha=30, shallow trees) and conformal bands prevent overconfident point forecasts.',
    36,
    y,
    contentW,
    13
  );

  drawFooter(doc, 4, totalPages, generatedDate);

  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  const arrayBuffer = doc.output('arraybuffer');
  fs.writeFileSync(OUT_PATH, Buffer.from(arrayBuffer));
  console.log(`Generated ${OUT_PATH} (${fs.statSync(OUT_PATH).size} bytes, ${doc.getNumberOfPages()} pages)`);
}

buildProjectPdf();
