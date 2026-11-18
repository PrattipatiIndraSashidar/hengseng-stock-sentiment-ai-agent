"""Agents 4 & 5: Financial Intelligence Agent & Feature Engineering Agent."""

from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from stock_sentiment_agent.collectors import PriceCollector
from stock_sentiment_agent.config import StockConfig, get_settings
from stock_sentiment_agent.database import DatabaseManager
from stock_sentiment_agent.preprocessing import FeatureEngineer, extract_explicit_trading_position


class FeatureEngineeringAgent:
    """Agent 5: Builds session-aligned, leakage-free technical + social + news feature matrices."""

    AGENT_NAME = "FeatureEngineeringAgent"

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.engineer = FeatureEngineer()

    def run(
        self,
        stock: StockConfig,
        price_bars: List[Dict[str, Any]],
        scored_social: List[Dict[str, Any]],
        scored_news: List[Dict[str, Any]],
        has_genuine_social: Optional[bool] = None,
        has_genuine_news: Optional[bool] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        if has_genuine_social is None:
            has_genuine_social = any(not p.get("is_simulated", False) for p in scored_social)
        if has_genuine_news is None:
            has_genuine_news = any(not a.get("is_simulated", False) for a in scored_news)

        feature_columns = self.engineer.select_feature_columns(
            has_genuine_social=bool(has_genuine_social),
            has_genuine_news=bool(has_genuine_news),
        )
        rows = self.engineer.build_features(
            ticker=stock.ticker,
            exchange=stock.exchange,
            price_bars=price_bars,
            scored_social_posts=scored_social,
            scored_news=scored_news,
        )
        feature_dicts = self.engineer.to_dicts(rows)
        self._persist_features(feature_dicts)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self.db.log_agent(
            agent_name=self.AGENT_NAME,
            status="SUCCESS",
            ticker=stock.ticker,
            message=(
                f"Engineered {len(feature_dicts)} session-aligned feature rows "
                f"({len(feature_columns)} model features; genuine_social={has_genuine_social}, genuine_news={has_genuine_news})."
            ),
            records_processed=len(feature_dicts),
            duration_ms=elapsed_ms,
        )
        return {
            "ticker": stock.ticker,
            "features": feature_dicts,
            "feature_columns": feature_columns,
            "has_genuine_social": bool(has_genuine_social),
            "has_genuine_news": bool(has_genuine_news),
        }

    def _persist_features(self, rows: List[Dict[str, Any]]) -> None:
        if self.db.sqlite_path is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self.db.sqlite_connection() as conn:
            for r in rows:
                conn.execute(
                    """
                    INSERT INTO financial_features (
                        ticker, trading_date, social_mentions, social_sentiment, pos_neg_ratio,
                        engagement_score, avg_author_followers, mention_growth, sentiment_change,
                        news_sentiment, news_count, news_relevance, news_recency,
                        rsi_14, macd, macd_signal, sma_5, sma_10, sma_20, volatility_10,
                        support_level, resistance_level, target_next_close, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, trading_date) DO UPDATE SET
                        social_mentions=excluded.social_mentions,
                        social_sentiment=excluded.social_sentiment,
                        news_sentiment=excluded.news_sentiment,
                        rsi_14=excluded.rsi_14,
                        macd=excluded.macd,
                        support_level=excluded.support_level,
                        resistance_level=excluded.resistance_level,
                        target_next_close=excluded.target_next_close
                    """,
                    (
                        r["ticker"], r["trading_date"], r["social_mentions"], r["social_sentiment"],
                        r["pos_neg_ratio"], r["engagement_score"], r["avg_author_followers"],
                        r["mention_growth"], r["sentiment_change"], r["news_sentiment"], r["news_count"],
                        r["news_relevance"], r["news_recency"], r["rsi_14"], r["macd"], r["macd_signal"],
                        r["sma_5"], r["sma_10"], r["sma_20"], r["volatility_10"],
                        r["support_level"], r["resistance_level"], r["target_next_close"], now,
                    ),
                )


class FinancialIntelligenceAgent:
    """Agent 4: Extracts all 20 required financial intelligence fields per ticker."""

    AGENT_NAME = "FinancialIntelligenceAgent"

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.settings = get_settings()
        self.price_collector = PriceCollector()

    def collect_prices(self, stock: StockConfig, allow_simulated_fallback: bool = False) -> Dict[str, Any]:
        bars, meta = self.price_collector.collect(
            stock,
            lookback_days=self.settings.lookback_days_price,
            allow_simulated_fallback=allow_simulated_fallback,
        )
        bar_dicts = [asdict(b) for b in bars]
        self._persist_prices(bar_dicts)
        return {"bars": bar_dicts, "meta": meta}

    def run(
        self,
        stock: StockConfig,
        feature_rows: List[Dict[str, Any]],
        sentiment_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        if not feature_rows:
            raise ValueError(f"No feature rows available for {stock.ticker}")

        latest = feature_rows[-1]
        social_sum = sentiment_result["social_summary"]
        news_sum = sentiment_result["news_summary"]
        combined_score = float(sentiment_result["combined_score"])

        # Determine Market Direction combining sentiment + momentum
        if combined_score > 0.12 or (combined_score > 0.04 and latest["close"] >= latest["sma_10"]):
            market_direction = "Bullish"
        elif combined_score < -0.12 or (combined_score < -0.04 and latest["close"] < latest["sma_10"]):
            market_direction = "Bearish"
        else:
            market_direction = "Neutral"

        all_texts = [p.get("post_text", "") for p in sentiment_result["scored_social"]] + [
            f"{a.get('headline', '')} {a.get('summary', '')}" for a in sentiment_result["scored_news"]
        ]
        trading_position = extract_explicit_trading_position(all_texts)

        top_news = sorted(
            sentiment_result["scored_news"],
            key=lambda a: float(a.get("relevance_score", 0.0)),
            reverse=True,
        )[:3]

        from stock_sentiment_agent.config import EXCHANGE_CONFIGS
        ex_cfg = EXCHANGE_CONFIGS[stock.exchange]

        intelligence = {
            "ticker": stock.ticker,
            "company_name": stock.company_name,
            "exchange": stock.exchange,
            "currency": ex_cfg.currency,
            "currency_symbol": ex_cfg.currency_symbol,
            "exchange_timezone": ex_cfg.timezone,
            "latest_price": latest["close"],
            "latest_trading_date": latest["trading_date"],
            "total_social_mentions": social_sum["total_items"],
            "positive_mentions": social_sum["positive_count"],
            "negative_mentions": social_sum["negative_count"],
            "neutral_mentions": social_sum["neutral_count"],
            "social_sentiment_score": social_sum["engagement_weighted_score"],
            "news_sentiment_score": news_sum["engagement_weighted_score"],
            "combined_sentiment_score": combined_score,
            "bullish_probability": sentiment_result["combined_summary"]["bullish_probability"],
            "bearish_probability": sentiment_result["combined_summary"]["bearish_probability"],
            "market_direction": market_direction,
            "mentioned_trading_position": trading_position,
            "support_levels": [round(latest["support_level"] * 0.985, 2), round(latest["support_level"], 2)],
            "resistance_levels": [round(latest["resistance_level"], 2), round(latest["resistance_level"] * 1.015, 2)],
            "rsi_14": latest["rsi_14"],
            "macd": latest["macd"],
            "macd_signal": latest["macd_signal"],
            "moving_averages": {
                "sma_5": latest["sma_5"],
                "sma_10": latest["sma_10"],
                "sma_20": latest["sma_20"],
            },
            "trading_volume": latest["volume"],
            "news_highlights": [
                {
                    "headline": n["headline"],
                    "source": n["source"],
                    "url": n["article_url"],
                    "sentiment": n["sentiment_label"],
                    "published_utc": n["published_utc"],
                }
                for n in top_news
            ],
        }

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self.db.log_agent(
            agent_name=self.AGENT_NAME,
            status="SUCCESS",
            ticker=stock.ticker,
            message=f"Extracted 20 intelligence fields; direction={market_direction}, position={trading_position}.",
            records_processed=1,
            duration_ms=elapsed_ms,
        )
        return intelligence

    def _persist_prices(self, bars: List[Dict[str, Any]]) -> None:
        if self.db.sqlite_path is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self.db.sqlite_connection() as conn:
            for b in bars:
                conn.execute(
                    """
                    INSERT INTO stock_prices (
                        ticker, trading_date, open, high, low, close, volume, daily_return, is_simulated, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, trading_date) DO UPDATE SET
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume,
                        daily_return=excluded.daily_return
                    """,
                    (
                        b["ticker"], b["trading_date"], b["open"], b["high"], b["low"],
                        b["close"], b["volume"], b["daily_return"], int(b.get("is_simulated", False)), now,
                    ),
                )
