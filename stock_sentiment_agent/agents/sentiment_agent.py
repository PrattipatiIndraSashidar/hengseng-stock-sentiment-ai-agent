"""Agent 3: Multilingual Sentiment Analysis Agent (Social + Financial News separately & combined)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from stock_sentiment_agent.config import StockConfig, get_settings
from stock_sentiment_agent.database import DatabaseManager
from stock_sentiment_agent.models import MultilingualSentimentAnalyzer
from stock_sentiment_agent.preprocessing import align_timestamp_to_trading_session


class SentimentAnalysisAgent:
    AGENT_NAME = "SentimentAnalysisAgent"

    def __init__(self, db: DatabaseManager, use_hf: bool = False) -> None:
        self.db = db
        settings = get_settings()
        self.analyzer = MultilingualSentimentAnalyzer(model_name=settings.sentiment_model_name, use_hf=use_hf)

    def run(
        self,
        stock: StockConfig,
        social_posts: List[Dict[str, Any]],
        news_articles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        started = time.perf_counter()

        scored_social: List[Dict[str, Any]] = []
        by_platform: Dict[str, List[Dict[str, Any]]] = {}

        for post in social_posts:
            pred = self.analyzer.analyze_text(str(post.get("post_text", "")))
            weight = self.analyzer.compute_engagement_weight(
                likes=int(post.get("likes") or 0),
                comments=int(post.get("comments") or 0),
                reposts=int(post.get("reposts") or 0),
                followers=post.get("author_follower_count"),
            )
            aligned = align_timestamp_to_trading_session(str(post.get("utc_timestamp", "")), stock.exchange)
            enriched = {
                **post,
                "aligned_trading_date": aligned,
                "sentiment_label": pred.label,
                "sentiment_score": pred.score,
                "confidence": pred.confidence,
                "engagement_weight": weight,
            }
            scored_social.append(enriched)
            by_platform.setdefault(str(post.get("source_platform", "Social")), []).append(enriched)

        scored_news: List[Dict[str, Any]] = []
        for art in news_articles:
            text = f"{art.get('headline', '')}. {art.get('summary', '')}"
            pred = self.analyzer.analyze_text(text)
            aligned = align_timestamp_to_trading_session(str(art.get("published_utc", "")), stock.exchange)
            scored_news.append(
                {
                    **art,
                    "aligned_trading_date": aligned,
                    "sentiment_label": pred.label,
                    "sentiment_score": pred.score,
                    "confidence": pred.confidence,
                    "engagement_weight": max(0.5, float(art.get("relevance_score", 0.8)) * 2.0),
                }
            )

        social_summary = self.analyzer.aggregate_scores(scored_social, use_engagement=True)
        news_summary = self.analyzer.aggregate_scores(scored_news, use_engagement=True)

        # Transparent 55% Social (engagement-weighted) + 45% Financial News (relevance-weighted) combination
        s_score = social_summary["engagement_weighted_score"] if scored_social else 0.0
        n_score = news_summary["engagement_weighted_score"] if scored_news else 0.0
        if scored_social and scored_news:
            combined_score = round(0.55 * s_score + 0.45 * n_score, 4)
        elif scored_social:
            combined_score = round(s_score, 4)
        else:
            combined_score = round(n_score, 4)

        combined_summary = self.analyzer.aggregate_scores(scored_social + scored_news, use_engagement=True)
        combined_summary["sentiment_score"] = combined_score

        platform_summaries = {
            plat: self.analyzer.aggregate_scores(items, use_engagement=True)
            for plat, items in by_platform.items()
        }

        self._persist_sentiment(stock.ticker, scored_news, social_summary, news_summary, combined_summary, platform_summaries)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self.db.log_agent(
            agent_name=self.AGENT_NAME,
            status="SUCCESS",
            ticker=stock.ticker,
            message=f"Scored {len(scored_social)} social + {len(scored_news)} news items via {self.analyzer.engine_name}.",
            records_processed=len(scored_social) + len(scored_news),
            duration_ms=elapsed_ms,
        )

        return {
            "ticker": stock.ticker,
            "engine": self.analyzer.engine_name,
            "scored_social": scored_social,
            "scored_news": scored_news,
            "social_summary": social_summary,
            "news_summary": news_summary,
            "combined_summary": combined_summary,
            "combined_score": combined_score,
            "platform_summaries": platform_summaries,
        }

    def _persist_sentiment(
        self,
        ticker: str,
        scored_news: List[Dict[str, Any]],
        social_summary: Dict[str, Any],
        news_summary: Dict[str, Any],
        combined_summary: Dict[str, Any],
        platform_summaries: Dict[str, Dict[str, Any]],
    ) -> None:
        if self.db.sqlite_path is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.db.sqlite_connection() as conn:
            for art in scored_news:
                conn.execute(
                    "UPDATE news_articles SET sentiment_label=?, sentiment_score=? WHERE article_url=?",
                    (art["sentiment_label"], art["sentiment_score"], art["article_url"]),
                )
            scopes = [("social", social_summary), ("news", news_summary), ("combined", combined_summary)]
            scopes.extend(platform_summaries.items())
            for scope, agg in scopes:
                conn.execute(
                    """
                    INSERT INTO sentiment_results (
                        ticker, trading_date, source_scope, total_items,
                        positive_count, negative_count, neutral_count,
                        positive_pct, negative_pct, neutral_pct,
                        sentiment_score, engagement_weighted_score,
                        bullish_probability, bearish_probability, confidence_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, trading_date, source_scope) DO UPDATE SET
                        total_items=excluded.total_items,
                        positive_count=excluded.positive_count,
                        negative_count=excluded.negative_count,
                        neutral_count=excluded.neutral_count,
                        sentiment_score=excluded.sentiment_score,
                        engagement_weighted_score=excluded.engagement_weighted_score,
                        bullish_probability=excluded.bullish_probability,
                        bearish_probability=excluded.bearish_probability,
                        confidence_score=excluded.confidence_score
                    """,
                    (
                        ticker, today, scope, agg["total_items"],
                        agg["positive_count"], agg["negative_count"], agg["neutral_count"],
                        agg["positive_pct"], agg["negative_pct"], agg["neutral_pct"],
                        agg["sentiment_score"], agg["engagement_weighted_score"],
                        agg["bullish_probability"], agg["bearish_probability"], agg["confidence_score"], now,
                    ),
                )
