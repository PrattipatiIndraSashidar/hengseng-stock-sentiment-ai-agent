"""Streamlit 7-Page Dark Financial Terminal Dashboard for HengSeng Sentiment Price Prediction Agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_PARENT = str(Path(__file__).resolve().parents[2])
if ROOT_PARENT not in sys.path:
    sys.path.insert(0, ROOT_PARENT)

from stock_sentiment_agent.agents import PipelineOrchestrator

st.set_page_config(
    page_title="HengSeng Sentiment Price Prediction Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #090D16; color: #E2E8F0; }
    [data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #1E293B; }
    .metric-card {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def load_pipeline_data(force_refresh: bool = False):
    snap = Path(__file__).resolve().parents[1] / "data" / "latest_pipeline_output.json"
    if snap.exists() and not force_refresh:
        return json.loads(snap.read_text(encoding="utf-8"))
    return PipelineOrchestrator().run_all()


st.sidebar.title("HengSeng Terminal")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Hong Kong Market",
        "South Korea Market",
        "Sentiment Analysis",
        "Stock Intelligence",
        "Price Prediction",
        "Reports",
    ],
)

if st.sidebar.button("Run Agent Pipeline"):
    st.cache_data.clear()
    data = load_pipeline_data(force_refresh=True)
else:
    data = load_pipeline_data()

summaries = data.get("summaries", [])
any_sim = any(s.get("data_provenance", {}).get("contains_simulated_data") for s in summaries)
if any_sim:
    st.info(
        "Demonstration Mode Notice: APIFY_API_TOKEN or EXA_API_KEY is not configured. "
        "Simulated sample social/news records are clearly labeled and separated from live market feeds."
    )

# Build flat dataframe for tables and charts
rows = []
for s in summaries:
    intel = s["intelligence"]
    pred = s["prediction"]
    rows.append(
        {
            "Ticker": intel["ticker"],
            "Company": intel["company_name"],
            "Exchange": intel["exchange"],
            "Price": intel["latest_price"],
            "Predicted n+1": pred["predicted_close"],
            "Pred % Change": pred["predicted_pct_change"],
            "Social Mentions": intel["total_social_mentions"],
            "Social Sentiment": intel["social_sentiment_score"],
            "News Sentiment": intel["news_sentiment_score"],
            "Combined Sentiment": intel["combined_sentiment_score"],
            "Bullish Prob": intel["bullish_probability"],
            "Bearish Prob": intel["bearish_probability"],
            "Direction": intel["market_direction"],
            "Position": intel["mentioned_trading_position"],
            "RSI (14)": intel["rsi_14"],
            "MACD": intel["macd"],
            "Best Model": pred["model_name"],
            "Val RMSE": pred.get("rmse", 0.0),
            "Val Dir Acc %": pred.get("directional_accuracy", 0.0),
        }
    )
df = pd.DataFrame(rows)

if page == "Overview":
    st.header("Market Overview — HKEX & KRX")
    total_posts = sum(s["intelligence"]["total_social_mentions"] for s in summaries)
    total_news = sum(len(s.get("recent_news", [])) for s in summaries)
    avg_sent = df["Combined Sentiment"].mean() if not df.empty else 0.0
    bullish_cnt = int((df["Direction"] == "Bullish").sum()) if not df.empty else 0
    bearish_cnt = int((df["Direction"] == "Bearish").sum()) if not df.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stocks Analyzed", len(summaries))
    c2.metric("Social Posts", total_posts)
    c3.metric("News Articles", total_news)
    c4.metric("Overall Sentiment", f"{avg_sent:+.3f}")
    c5.metric("Bullish / Bearish", f"{bullish_cnt} / {bearish_cnt}")

    fig = px.bar(
        df,
        x="Ticker",
        y="Combined Sentiment",
        color="Direction",
        color_discrete_map={"Bullish": "#10B981", "Bearish": "#EF4444", "Neutral": "#F59E0B"},
        title="Combined Multilingual Sentiment Score by Equity",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

elif page in ("Hong Kong Market", "South Korea Market"):
    ex = "HKEX" if page == "Hong Kong Market" else "KRX"
    st.header(f"{page} ({ex})")
    sub = df[df["Exchange"] == ex].sort_values("Combined Sentiment", ascending=False)
    st.dataframe(sub, use_container_width=True)
    fig = px.scatter(
        sub,
        x="Social Mentions",
        y="Pred % Change",
        size="Bullish Prob",
        color="Direction",
        text="Ticker",
        color_discrete_map={"Bullish": "#10B981", "Bearish": "#EF4444", "Neutral": "#F59E0B"},
        title=f"{ex} Mentions vs Predicted Next-Day Return (%)",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Sentiment Analysis":
    st.header("Multilingual Social & News Sentiment Analysis")
    plat_rows = []
    for s in summaries:
        t = s["intelligence"]["ticker"]
        for plat, psum in s["sentiment"]["platform_summaries"].items():
            plat_rows.append(
                {
                    "Ticker": t,
                    "Platform": plat,
                    "Items": psum["total_items"],
                    "Positive %": psum["positive_pct"],
                    "Negative %": psum["negative_pct"],
                    "Neutral %": psum["neutral_pct"],
                    "Engagement Score": psum["engagement_weighted_score"],
                }
            )
    pdf = pd.DataFrame(plat_rows)
    fig = px.bar(
        pdf,
        x="Ticker",
        y="Engagement Score",
        color="Platform",
        barmode="group",
        title="Platform-Wise Engagement-Weighted Sentiment",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pdf, use_container_width=True)

elif page == "Stock Intelligence":
    st.header("Stock Intelligence Terminal")
    tickers = [s["intelligence"]["ticker"] for s in summaries]
    selected = st.selectbox("Select Ticker", tickers)
    item = next(s for s in summaries if s["intelligence"]["ticker"] == selected)
    intel = item["intelligence"]
    pred = item["prediction"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", f"{intel['latest_price']:.2f}")
    c2.metric("Predicted n+1", f"{pred['predicted_close']:.2f}", f"{pred['predicted_pct_change']:+.2f}%")
    c3.metric("Combined Sentiment", f"{intel['combined_sentiment_score']:+.3f}")
    c4.metric("Direction / Position", f"{intel['market_direction']} · {intel['mentioned_trading_position']}")

    hist = pd.DataFrame(item.get("price_history", []))
    if not hist.empty:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=hist["trading_date"],
                    open=hist["open"],
                    high=hist["high"],
                    low=hist["low"],
                    close=hist["close"],
                    name=selected,
                )
            ]
        )
        fig.add_hline(y=intel["support_levels"][1], line_dash="dot", line_color="#10B981", annotation_text="Support")
        fig.add_hline(y=intel["resistance_levels"][0], line_dash="dot", line_color="#EF4444", annotation_text="Resistance")
        fig.update_layout(template="plotly_dark", title=f"{selected} Price Action with Support & Resistance")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Technical Indicators & News Highlights")
    st.json({"RSI_14": intel["rsi_14"], "MACD": intel["macd"], "Moving_Averages": intel["moving_averages"]})
    st.table(pd.DataFrame(intel["news_highlights"]))

elif page == "Price Prediction":
    st.header("Out-of-Sample Model Validation & n+1 Forecasts")
    tickers = [s["intelligence"]["ticker"] for s in summaries]
    selected = st.selectbox("Inspect Model Validation", tickers)
    item = next(s for s in summaries if s["intelligence"]["ticker"] == selected)

    val_df = pd.DataFrame(
        [
            {
                "Model": v["model_name"],
                "Best": "Selected" if v["is_best_model"] else "",
                "MAE": v["mae"],
                "RMSE": v["rmse"],
                "MAPE %": v["mape"],
                "Directional Acc %": v["directional_accuracy"],
            }
            for v in item["validation"]
        ]
    )
    st.dataframe(val_df, use_container_width=True)

    best_val = next((v for v in item["validation"] if v["is_best_model"]), item["validation"][-1])
    pred_hist = pd.DataFrame(best_val["predictions"])
    if not pred_hist.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_hist["prediction_date"], y=pred_hist["actual_close"], name="Actual Close"))
        fig.add_trace(go.Scatter(x=pred_hist["prediction_date"], y=pred_hist["predicted_close"], name="Predicted Close"))
        fig.update_layout(template="plotly_dark", title=f"{selected} Out-of-Sample Actual vs Predicted ({best_val['model_name']})")
        st.plotly_chart(fig, use_container_width=True)

    fi = best_val.get("feature_importances", {})
    if fi:
        fi_df = pd.DataFrame([{"Feature": k, "Importance": v} for k, v in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]])
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Top 10 Feature Importances", template="plotly_dark")
        st.plotly_chart(fig_fi, use_container_width=True)

elif page == "Reports":
    st.header("Exchange-Wise Reports & Exports")
    ex_filter = st.selectbox("Filter Exchange", ["ALL", "HKEX", "KRX"])
    filtered = df if ex_filter == "ALL" else df[df["Exchange"] == ex_filter]
    st.dataframe(filtered, use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV Report", data=csv_bytes, file_name="hengseng_report.csv", mime="text/csv")

    pdf_path = Path(data.get("report", {}).get("pdf_path", ""))
    if pdf_path.exists():
        st.download_button("Download PDF Report", data=pdf_path.read_bytes(), file_name="hengseng_report.pdf", mime="application/pdf")
