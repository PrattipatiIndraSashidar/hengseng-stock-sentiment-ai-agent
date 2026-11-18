# HengSeng Sentiment Price Prediction Agent — Live Data Integration & Final Assessment Report

- **Generated UTC:** `2026-11-17T09:44:49.722052+00:00`
- **Stocks Successfully Analysed:** `10` (5 HKEX · 5 KRX)

## 1. Social Media Sources & Connection Status (Apify)

| Platform | Exchange | Apify Actor | Status | Genuine Posts | Details |
|---|---|---|---|---:|---|
| LIHKG | HKEX | `apify/cheerio-scraper` | `unavailable` | 0 | LIHKG unavailable (Apify run start failed for apify/cheerio-scraper: HTTP 403: Forbidden - This Actor requires full access to your account. You must approve its permissions before running it: https://console.apify.com/actors/YrQuEkowkNCLdk4j2?approvePermissions=true). Loaded labeled SIMULATED DEMO DATA. |
| Telegram | HKEX | `apify/cheerio-scraper` | `unavailable` | 0 | Telegram actor unavailable (Apify run start failed for apify/cheerio-scraper: HTTP 403: Forbidden - This Actor requires full access to your account. You must approve its permissions before running it: https://console.apify.com/actors/YrQuEkowkNCLdk4j2?approvePermissions=true); using labeled SIMULATED DEMO DATA. |
| Naver Finance | KRX | `apify/cheerio-scraper` | `unavailable` | 0 | Naver Finance unavailable (Apify run start failed for apify/cheerio-scraper: HTTP 403: Forbidden - This Actor requires full access to your account. You must approve its permissions before running it: https://console.apify.com/actors/YrQuEkowkNCLdk4j2?approvePermissions=true); loaded labeled SIMULATED DEMO DATA. |
| Daum Finance | KRX | `apify/web-scraper` | `unavailable` | 0 | Daum Finance unavailable (Apify run start failed for apify/web-scraper: HTTP 403: Forbidden - This Actor requires full access to your account. You must approve its permissions before running it: https://console.apify.com/actors/moJRLRc85AitArpNN?approvePermissions=true); loaded labeled SIMULATED DEMO DATA. |

## 2. Exa Search API Connection Status

- **Status:** `connected`
- **Data Category:** `LIVE DATA`
- **Genuine Articles Collected:** `78`
- **Details:** Exa Search API authenticated and responding.

## 3. Market Data Provider Status

- **Status:** `connected`
- **Providers Used:** Yahoo Finance (query2.finance.yahoo.com)
- **Equities with Genuine Historical OHLCV + Live Latest Session:** `10/10`
- **Provenance:** Historical Series = `HISTORICAL DATA`, Latest Session Quote = `LIVE DATA`
- **Rate-Limit Mitigation:** CookieJar session handshake (`fc.yahoo.com`) + `query2`/`query1` failover + 0.35s throttling + 4h local JSON cache + Naver Finance KRX backup.

## 4 & 5. Collected Data Totals (Strict Genuine vs Demo Separation)

- **Genuine Social Posts Collected:** `0` *(Simulated demo posts excluded from ML training: 56)*
- **Genuine News Articles Collected:** `78` *(Simulated demo articles excluded from ML training: 0)*

## 6 & 7. Model Validation Results (Chronological Walk-Forward vs Naive Previous-Close Baseline)

- **Aggregate Best-Model Directional Accuracy:** `52.97%` (Naive Baseline mean: `46.62%`)
- **Stocks Where Best ML Model Beats/Matches Naive RMSE:** `6/10`

| Ticker | Best Model | MAE | RMSE | MAPE | Dir Acc | Naive RMSE | Validation Mode |
|---|---|---:|---:|---:|---:|---:|---|
| 0700.HK | XGBoost Regressor | 7.845 | 10.133 | 1.96% | 43.5% | 9.923 | Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV) |
| 9988.HK | Linear Regression | 2.435 | 3.216 | 2.45% | 60.9% | 3.482 | Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV) |
| 3690.HK | XGBoost Regressor | 1.754 | 2.204 | 2.38% | 39.1% | 2.213 | Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV) |
| 1810.HK | XGBoost Regressor | 0.697 | 0.900 | 2.46% | 34.8% | 0.874 | Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV) |
| 1299.HK | XGBoost Regressor | 0.843 | 1.174 | 1.12% | 69.6% | 1.182 | Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV) |
| 005930.KS | Linear Regression | 6644.017 | 7740.098 | 3.13% | 59.1% | 8849.961 | Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV) |
| 000660.KS | Linear Regression | 60333.743 | 71093.699 | 4.72% | 63.6% | 73772.993 | Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV) |
| 035420.KS | Random Forest Regressor | 4134.308 | 5688.395 | 2.29% | 40.9% | 5480.046 | Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV) |
| 005380.KS | Random Forest Regressor | 6228.592 | 7367.885 | 2.15% | 63.6% | 7978.665 | Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV) |
| 035720.KS | XGBoost Regressor | 512.976 | 821.645 | 1.65% | 54.5% | 786.607 | Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV) |

## 8. Remaining Limitations

- APIFY_API_TOKEN is missing or social actors could not reach live platforms; 0 genuine social posts were collected (labeled SIMULATED DEMO DATA is used only for UI demonstration and excluded from ML training).
- LIHKG enforces Cloudflare anti-bot protection and Daum Finance uses SPA client rendering, which can restrict Apify cheerio/web crawls even when APIFY_API_TOKEN is configured.
- transformers/PyTorch runtime is not installed in the container environment; multilingual sentiment uses weighted multilingual financial lexicon scoring rather than transformer inference.
