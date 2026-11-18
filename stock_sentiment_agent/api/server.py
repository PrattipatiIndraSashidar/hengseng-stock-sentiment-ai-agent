"""FastAPI backend exposing HengSeng Sentiment Price Prediction endpoints."""

from __future__ import annotations

import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stock_sentiment_agent.agents import PipelineOrchestrator
from stock_sentiment_agent.config import StockConfig, get_settings
from stock_sentiment_agent.database import get_db_manager

app = FastAPI(
    title="HengSeng Sentiment Price Prediction Agent API",
    version="1.0.0",
    description="Multilingual sentiment analysis and next-day price prediction for HKEX & KRX equities.",
)


class TickerUpdateRequest(BaseModel):
    ticker: str
    company_name: str
    exchange: str
    sector: str = "Technology"
    aliases: List[str] = []
    enabled: bool = True


@app.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "apify_configured": settings.has_apify,
        "exa_configured": settings.has_exa,
        "stocks_count": len(settings.stocks),
    }


@app.post("/pipeline/run")
def run_pipeline(ticker: Optional[str] = None):
    orchestrator = PipelineOrchestrator()
    if ticker:
        match = [s for s in get_settings().stocks if s.ticker == ticker]
        if not match:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found.")
        return orchestrator.run_single(match[0])
    return orchestrator.run_all()


@app.get("/dashboard/latest")
def get_latest_dashboard():
    settings = get_settings()
    snap = settings.data_dir / "latest_pipeline_output.json"
    if not snap.exists():
        return PipelineOrchestrator().run_all()
    return json.loads(snap.read_text(encoding="utf-8"))


@app.get("/stocks")
def list_stocks():
    return [
        {
            "ticker": s.ticker,
            "company_name": s.company_name,
            "exchange": s.exchange,
            "sector": s.sector,
            "aliases": s.aliases,
            "enabled": s.enabled,
        }
        for s in get_settings().stocks
    ]


@app.post("/stocks/configure")
def configure_stock(item: TickerUpdateRequest):
    settings = get_settings()
    stocks = [s for s in settings.stocks if s.ticker != item.ticker]
    stocks.append(StockConfig(**item.model_dump()))
    settings.save_stocks(stocks)
    return {"status": "updated", "stocks_count": len(stocks)}


@app.get("/logs")
def get_agent_logs(limit: int = 50):
    db = get_db_manager()
    return db.query_rows(
        "SELECT agent_name, status, ticker, message, records_processed, duration_ms, created_at "
        "FROM agent_logs ORDER BY id DESC LIMIT ?",
        (limit,),
    )


@app.get("/reports/csv")
def download_csv():
    path = get_settings().reports_dir / "hengseng_report.csv"
    if not path.exists():
        PipelineOrchestrator().run_all()
    return FileResponse(path, media_type="text/csv", filename="hengseng_report.csv")


@app.get("/reports/pdf")
def download_pdf():
    path = get_settings().reports_dir / "hengseng_report.pdf"
    if not path.exists():
        PipelineOrchestrator().run_all()
    return FileResponse(path, media_type="application/pdf", filename="hengseng_report.pdf")
