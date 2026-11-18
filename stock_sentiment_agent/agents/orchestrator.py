"""Central Multi-Agent Pipeline Orchestrator with 3-tier Data Provenance & Connection Diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from stock_sentiment_agent.agents.financial_agent import FeatureEngineeringAgent, FinancialIntelligenceAgent
from stock_sentiment_agent.agents.news_agent import FinancialNewsResearchAgent
from stock_sentiment_agent.agents.prediction_agent import PricePredictionAgent
from stock_sentiment_agent.agents.reporting_agent import ReportGenerationAgent
from stock_sentiment_agent.agents.sentiment_agent import SentimentAnalysisAgent
from stock_sentiment_agent.agents.social_agent import SocialDataCollectionAgent
from stock_sentiment_agent.agents.validation_agent import ModelValidationAgent
from stock_sentiment_agent.collectors.apify_collector import (
    DATA_CAT_HISTORICAL,
    DATA_CAT_LIVE,
    DATA_CAT_SIMULATED,
)
from stock_sentiment_agent.config import StockConfig, get_settings
from stock_sentiment_agent.database import DatabaseManager, get_db_manager


class PipelineOrchestrator:
    """Coordinates all 8 independent agents in deterministic sequence."""

    def __init__(self, db: Optional[DatabaseManager] = None, use_hf: bool = False) -> None:
        self.settings = get_settings(reload=True)
        self.db = db or get_db_manager()
        self.db.init_tables()
        self.db.seed_stocks(self.settings.stocks)

        self.social_agent = SocialDataCollectionAgent(self.db)
        self.news_agent = FinancialNewsResearchAgent(self.db)
        self.sentiment_agent = SentimentAnalysisAgent(self.db, use_hf=use_hf)
        self.financial_agent = FinancialIntelligenceAgent(self.db)
        self.feature_agent = FeatureEngineeringAgent(self.db)
        self.validation_agent = ModelValidationAgent(self.db)
        self.prediction_agent = PricePredictionAgent(self.db)
        self.reporting_agent = ReportGenerationAgent(self.db)

    def run_all(self, stocks: Optional[List[StockConfig]] = None) -> Dict[str, Any]:
        target_stocks = stocks or self.settings.stocks
        self.db.seed_stocks(target_stocks)

        from stock_sentiment_agent.collectors.apify_collector import ApifyBaseCollector
        from stock_sentiment_agent.collectors.exa_collector import ExaNewsCollector

        apify_probe = ApifyBaseCollector(self.settings.apify_api_token).probe_sources()
        exa_ok, exa_reason = ExaNewsCollector(self.settings.exa_api_key).test_connection()
        exa_probe = {"connected": exa_ok, "reason": exa_reason}

        # Execute 1 batched scrape per platform across all regional tickers
        self.social_agent.prefetch_all(target_stocks)

        stock_summaries: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        for stock in target_stocks:
            try:
                summary = self.run_single(stock)
                stock_summaries.append(summary)
            except Exception as exc:
                errors.append({"ticker": stock.ticker, "error": str(exc)})
                self.db.log_agent("PipelineOrchestrator", "ERROR", str(exc), ticker=stock.ticker)

        source_status = self._summarize_source_status(stock_summaries)
        report_bundle = (
            self.reporting_agent.run(
                stock_summaries,
                api_connection_status=source_status,
                apify_probe=apify_probe,
                exa_probe=exa_probe,
            )
            if stock_summaries
            else {}
        )
        assessment_summary = report_bundle.get("assessment_summary")

        result = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "stocks_analyzed": len(stock_summaries),
            "configuration_validation": self.settings.validate_configuration(),
            "api_connection_status": source_status,
            "assessment_summary": assessment_summary,
            "summaries": stock_summaries,
            "report": {k: v for k, v in report_bundle.items() if k != "assessment_summary"},
            "errors": errors,
        }
        snapshot_path = self.settings.data_dir / "latest_pipeline_output.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def run_single(self, stock: StockConfig) -> Dict[str, Any]:
        social_out = self.social_agent.run(stock)
        news_out = self.news_agent.run(stock)
        sentiment_out = self.sentiment_agent.run(stock, social_out["posts"], news_out["articles"])

        # Strict ML isolation: ONLY genuine (non-simulated) records may enter ML training features
        genuine_scored_social = [p for p in sentiment_out["scored_social"] if not p.get("is_simulated")]
        genuine_scored_news = [a for a in sentiment_out["scored_news"] if not a.get("is_simulated")]

        genuine_posts_count = len(genuine_scored_social)
        simulated_posts_count = len(sentiment_out["scored_social"]) - genuine_posts_count
        genuine_news_count = len(genuine_scored_news)
        simulated_news_count = len(sentiment_out["scored_news"]) - genuine_news_count

        has_genuine_social = genuine_posts_count > 0
        has_genuine_news = genuine_news_count > 0

        price_out = self.financial_agent.collect_prices(stock, allow_simulated_fallback=False)
        features_out = self.feature_agent.run(
            stock=stock,
            price_bars=price_out["bars"],
            scored_social=genuine_scored_social,
            scored_news=genuine_scored_news,
            has_genuine_social=has_genuine_social,
            has_genuine_news=has_genuine_news,
        )
        intelligence = self.financial_agent.run(stock, features_out["features"], sentiment_out)
        validation_results = self.validation_agent.run(
            stock=stock,
            feature_rows=features_out["features"],
            feature_columns=features_out["feature_columns"],
        )
        prediction = self.prediction_agent.run(
            stock=stock,
            feature_rows=features_out["features"],
            feature_columns=features_out["feature_columns"],
            validation_results=validation_results,
        )

        social_cat = DATA_CAT_LIVE if has_genuine_social else DATA_CAT_SIMULATED
        news_cat = DATA_CAT_LIVE if has_genuine_news else news_out["meta"].get("data_category", DATA_CAT_SIMULATED)
        price_cat = price_out["meta"].get("data_category", DATA_CAT_HISTORICAL)
        latest_quote_cat = price_out["meta"].get("latest_quote_category", DATA_CAT_LIVE)

        if price_cat == DATA_CAT_SIMULATED:
            validation_mode = "Demo-Model Validation (Simulated Prices)"
            is_demo_model = True
        elif has_genuine_social or has_genuine_news:
            validation_mode = "Genuine Multimodal Walk-Forward (Live Sentiment + Historical OHLCV)"
            is_demo_model = False
        else:
            validation_mode = "Genuine Out-of-Sample Walk-Forward (Historical OHLCV Technicals; Live Social/News Unavailable)"
            is_demo_model = False

        # Sort recent_posts so verified live posts appear before labeled demo placeholders
        ordered_posts = sorted(
            sentiment_out["scored_social"],
            key=lambda p: (bool(p.get("is_simulated")), -float(p.get("engagement_weight", 1.0))),
        )

        platform_post_counts: Dict[str, Dict[str, int]] = {}
        for p in sentiment_out["scored_social"]:
            plat = str(p.get("source_platform") or "Unknown")
            entry = platform_post_counts.setdefault(plat, {"genuine": 0, "simulated": 0})
            if p.get("is_simulated"):
                entry["simulated"] += 1
            else:
                entry["genuine"] += 1

        return {
            "stock": asdict(stock),
            "intelligence": intelligence,
            "prediction": prediction,
            "validation": [asdict(v) for v in validation_results],
            "sentiment": {
                "engine": sentiment_out["engine"],
                "social_summary": sentiment_out["social_summary"],
                "news_summary": sentiment_out["news_summary"],
                "combined_summary": sentiment_out["combined_summary"],
                "platform_summaries": sentiment_out["platform_summaries"],
            },
            "data_provenance": {
                "social": social_out["platform_notes"],
                "news": news_out["meta"],
                "prices": price_out["meta"],
                "categories": {
                    "prices": price_cat,
                    "latest_quote": latest_quote_cat,
                    "social": social_cat,
                    "news": news_cat,
                },
                "platform_post_counts": platform_post_counts,
                "genuine_posts_count": genuine_posts_count,
                "simulated_posts_count": simulated_posts_count,
                "genuine_news_count": genuine_news_count,
                "simulated_news_count": simulated_news_count,
                "features_used": features_out["feature_columns"],
                "validation_mode": validation_mode,
                "is_demo_model": is_demo_model,
                "contains_simulated_data": bool(
                    simulated_posts_count > 0
                    or simulated_news_count > 0
                    or price_cat == DATA_CAT_SIMULATED
                ),
            },
            "recent_posts": ordered_posts[:12],
            "recent_news": sentiment_out["scored_news"][:6],
            "price_history": price_out["bars"][-30:],
        }

    def _summarize_source_status(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources: Dict[str, Dict[str, Any]] = {}
        providers_seen = set()
        for s in summaries:
            prov = s.get("data_provenance", {})
            for plat, meta in prov.get("social", {}).items():
                key = f"Apify · {plat}"
                prev = sources.get(key)
                # Prefer connected / live status across multiple tickers of same exchange
                if not prev or (prev.get("data_category") != DATA_CAT_LIVE and meta.get("data_category") == DATA_CAT_LIVE):
                    sources[key] = {
                        "source": key,
                        "kind": "Social Media",
                        "status": meta.get("status", "unavailable"),
                        "data_category": meta.get("data_category", DATA_CAT_SIMULATED),
                        "reason": meta.get("reason", ""),
                    }
            news_meta = prov.get("news", {})
            prev_news = sources.get("Exa Search API")
            if not prev_news or (prev_news.get("status") != "connected" and news_meta.get("status") == "connected"):
                sources["Exa Search API"] = {
                    "source": "Exa Search API",
                    "kind": "Financial News",
                    "status": news_meta.get("status", "unavailable"),
                    "data_category": news_meta.get("data_category", DATA_CAT_SIMULATED),
                    "reason": news_meta.get("reason", ""),
                }
            price_meta = prov.get("prices", {})
            if price_meta.get("provider"):
                providers_seen.add(str(price_meta["provider"]))
            existing_price = sources.get("Market Prices (Yahoo Finance / Naver)")
            if not existing_price or price_meta.get("status") == "connected":
                sources["Market Prices (Yahoo Finance / Naver)"] = {
                    "source": "Market Prices (Yahoo Finance / Naver)",
                    "kind": "Market Prices",
                    "status": price_meta.get("status", "unavailable"),
                    "data_category": price_meta.get("data_category", DATA_CAT_HISTORICAL),
                    "reason": (
                        f"Active provider(s): {', '.join(sorted(providers_seen)) or 'Yahoo Finance'}. "
                        f"{price_meta.get('reason', '')}"
                    ).strip(),
                }

        sources["SQLite / SQLAlchemy DB"] = {
            "source": "SQLite / SQLAlchemy DB",
            "kind": "Persistence",
            "status": "connected",
            "data_category": "LOCAL DATABASE",
            "reason": f"Connected to {'SQLite fallback' if self.settings.using_sqlite_fallback else 'configured SQL database'} (9 tables).",
        }
        return list(sources.values())
