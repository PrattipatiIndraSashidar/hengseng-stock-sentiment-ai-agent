"""Agent 2: Financial News Research Agent via Exa Search API."""

from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from stock_sentiment_agent.collectors import ExaNewsCollector, NormalizedNewsArticle
from stock_sentiment_agent.config import StockConfig, get_settings
from stock_sentiment_agent.database import DatabaseManager
from stock_sentiment_agent.preprocessing import align_timestamp_to_trading_session


class FinancialNewsResearchAgent:
    AGENT_NAME = "FinancialNewsResearchAgent"

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.settings = get_settings()
        self.collector = ExaNewsCollector()

    def run(self, stock: StockConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        articles, meta = self.collector.collect(stock, lookback_days=self.settings.lookback_days_social)
        self._persist_articles(stock, articles)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        is_simulated = any(a.is_simulated for a in articles)
        self.db.log_agent(
            agent_name=self.AGENT_NAME,
            status="SUCCESS",
            ticker=stock.ticker,
            message=f"Collected {len(articles)} news articles ({meta.get('mode')}).",
            records_processed=len(articles),
            duration_ms=elapsed_ms,
        )
        return {
            "ticker": stock.ticker,
            "articles": [asdict(a) for a in articles],
            "meta": meta,
            "is_simulated": is_simulated,
        }

    def _persist_articles(self, stock: StockConfig, articles: List[NormalizedNewsArticle]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if self.db.sqlite_path is None:
            return
        with self.db.sqlite_connection() as conn:
            for a in articles:
                aligned = align_timestamp_to_trading_session(a.published_utc, stock.exchange)
                conn.execute(
                    """
                    INSERT INTO news_articles (
                        ticker, headline, article_url, published_at, published_utc,
                        aligned_trading_date, summary, source, relevance_score,
                        is_simulated, collected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(article_url) DO UPDATE SET
                        headline=excluded.headline,
                        summary=excluded.summary,
                        relevance_score=excluded.relevance_score,
                        aligned_trading_date=excluded.aligned_trading_date
                    """,
                    (
                        a.ticker, a.headline, a.article_url, a.published_at, a.published_utc,
                        aligned, a.summary, a.source, a.relevance_score, int(a.is_simulated), now,
                    ),
                )
