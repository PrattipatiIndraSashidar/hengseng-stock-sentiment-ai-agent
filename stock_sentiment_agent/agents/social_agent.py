"""Agent 1: Social Data Collection Agent (LIHKG, Telegram, Naver Finance, Daum Finance)."""

from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from stock_sentiment_agent.collectors import (
    ApifyBaseCollector,
    DaumFinanceCollector,
    LihkgCollector,
    NaverFinanceCollector,
    NormalizedSocialPost,
    TelegramCollector,
)
from stock_sentiment_agent.config import StockConfig, get_settings
from stock_sentiment_agent.database import DatabaseManager
from stock_sentiment_agent.preprocessing import align_timestamp_to_trading_session, clean_text


class SocialDataCollectionAgent:
    AGENT_NAME = "SocialDataCollectionAgent"

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.settings = get_settings()
        self.lihkg = LihkgCollector()
        self.telegram = TelegramCollector()
        self.naver = NaverFinanceCollector()
        self.daum = DaumFinanceCollector()
        self._prefetched: Dict[Tuple[str, str], Tuple[List[NormalizedSocialPost], Dict[str, str]]] = {}

    def prefetch_all(self, stocks: List[StockConfig]) -> None:
        """Run 1 batched Apify/direct crawl per platform across all regional stocks."""
        self._prefetched.clear()
        lookback = self.settings.lookback_days_social
        hkex_stocks = [s for s in stocks if s.exchange == "HKEX"]
        krx_stocks = [s for s in stocks if s.exchange == "KRX"]

        if hkex_stocks:
            for ticker, res in self.lihkg.collect_all(hkex_stocks, lookback_days=lookback).items():
                self._prefetched[(self.lihkg.PLATFORM, ticker)] = res
            for ticker, res in self.telegram.collect_all(hkex_stocks, lookback_days=lookback).items():
                self._prefetched[(self.telegram.PLATFORM, ticker)] = res

        if krx_stocks:
            for ticker, res in self.naver.collect_all(krx_stocks, lookback_days=lookback).items():
                self._prefetched[(self.naver.PLATFORM, ticker)] = res
            for ticker, res in self.daum.collect_all(krx_stocks, lookback_days=lookback).items():
                self._prefetched[(self.daum.PLATFORM, ticker)] = res

    def run(self, stock: StockConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        collectors = (
            [self.lihkg, self.telegram]
            if stock.exchange == "HKEX"
            else [self.naver, self.daum]
        )

        collected: List[NormalizedSocialPost] = []
        platform_notes: Dict[str, Dict[str, str]] = {}

        for collector in collectors:
            try:
                key = (collector.PLATFORM, stock.ticker)
                if key in self._prefetched:
                    posts, meta = self._prefetched[key]
                else:
                    posts, meta = collector.collect(stock, lookback_days=self.settings.lookback_days_social)
                platform_notes[collector.PLATFORM] = meta
                ApifyBaseCollector.save_raw_snapshot(collector.PLATFORM, stock.ticker, posts, self.settings.data_dir)
                collected.extend(posts)
            except Exception as exc:
                platform_notes[collector.PLATFORM] = {"status": "unavailable", "mode": "error", "reason": str(exc)}

        deduped = ApifyBaseCollector.deduplicate_posts(collected)
        self._persist_posts(stock, deduped)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        genuine_count = sum(1 for p in deduped if not p.is_simulated)
        sim_count = sum(1 for p in deduped if p.is_simulated)
        is_simulated = genuine_count == 0
        self.db.log_agent(
            agent_name=self.AGENT_NAME,
            status="SUCCESS",
            ticker=stock.ticker,
            message=f"Collected {genuine_count} genuine + {sim_count} labeled demo social posts.",
            records_processed=len(deduped),
            duration_ms=elapsed_ms,
        )

        return {
            "ticker": stock.ticker,
            "posts": [asdict(p) for p in deduped],
            "platform_notes": platform_notes,
            "genuine_count": genuine_count,
            "simulated_count": sim_count,
            "is_simulated": is_simulated,
        }

    def _persist_posts(self, stock: StockConfig, posts: List[NormalizedSocialPost]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if self.db.sqlite_path is None:
            return
        with self.db.sqlite_connection() as conn:
            for p in posts:
                aligned = align_timestamp_to_trading_session(p.utc_timestamp, stock.exchange)
                cleaned = clean_text(p.post_text)
                conn.execute(
                    """
                    INSERT INTO social_posts (
                        ticker, company_name, post_id, post_text, cleaned_text,
                        post_timestamp, utc_timestamp, aligned_trading_date,
                        source_platform, author_username, author_follower_count,
                        likes, comments, reposts, views, post_url, raw_payload,
                        is_simulated, collected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_platform, post_id) DO UPDATE SET
                        cleaned_text=excluded.cleaned_text,
                        aligned_trading_date=excluded.aligned_trading_date,
                        likes=excluded.likes,
                        comments=excluded.comments,
                        reposts=excluded.reposts,
                        views=excluded.views,
                        is_simulated=excluded.is_simulated
                    """,
                    (
                        p.ticker, p.company_name, p.post_id, p.post_text, cleaned,
                        p.post_timestamp, p.utc_timestamp, aligned,
                        p.source_platform, p.author_username, p.author_follower_count,
                        p.likes, p.comments, p.reposts, p.views, p.post_url, p.raw_payload,
                        int(p.is_simulated), now,
                    ),
                )
