"""Comprehensive test suite for HengSeng Sentiment Price Prediction Agent.

Runnable with either `pytest` or Python stdlib `python3 -m unittest`.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stock_sentiment_agent.agents import PipelineOrchestrator
from stock_sentiment_agent.collectors import ExaNewsCollector, LihkgCollector
from stock_sentiment_agent.config import StockConfig
from stock_sentiment_agent.database import DatabaseManager
from stock_sentiment_agent.models import ModelEvaluator, MultilingualSentimentAnalyzer
from stock_sentiment_agent.preprocessing import (
    FeatureEngineer,
    align_timestamp_to_trading_session,
    clean_text,
    extract_explicit_trading_position,
)


class TestHengSengPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_hengseng.db"
        self.db = DatabaseManager(f"sqlite:///{self.db_path}")
        self.db.init_tables()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_api_integration_missing_keys_uses_labeled_demo(self) -> None:
        stock = StockConfig("0700.HK", "Tencent", "HKEX", "Communication Services")
        lihkg = LihkgCollector(api_token="")
        posts, meta = lihkg.collect(stock)
        self.assertEqual(meta["mode"], "simulated_demo")
        self.assertTrue(len(posts) > 0)
        self.assertTrue(all(p.is_simulated for p in posts))

        exa = ExaNewsCollector(api_key="")
        news, nmeta = exa.collect(stock)
        self.assertEqual(nmeta["mode"], "simulated_demo")
        self.assertTrue(len(news) > 0)
        self.assertTrue(all(n.is_simulated for n in news))

    def test_social_preprocessing_and_explicit_position(self) -> None:
        raw = "Check <b>Tencent</b> https://example.com 業績亮眼, going long!   "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Check Tencent 業績亮眼, going long!")
        self.assertEqual(extract_explicit_trading_position([raw]), "Long")
        self.assertEqual(extract_explicit_trading_position(["Tencent earnings beat expectations."]), "Not mentioned")

    def test_multilingual_sentiment_and_engagement_weighting(self) -> None:
        analyzer = MultilingualSentimentAnalyzer(use_hf=False)
        pos_en = analyzer.analyze_text("Strong earnings beat and bullish revenue growth!")
        pos_zh = analyzer.analyze_text("業績亮眼，盈利增長超預期，突破看漲")
        neg_ko = analyzer.analyze_text("실적 부진 및 공매도 급증으로 하락 우려")
        self.assertEqual(pos_en.label, "Positive")
        self.assertEqual(pos_zh.label, "Positive")
        self.assertEqual(neg_ko.label, "Negative")

        w_low = analyzer.compute_engagement_weight(likes=1, comments=0, reposts=0)
        w_high = analyzer.compute_engagement_weight(likes=200, comments=40, reposts=20, followers=50000)
        self.assertGreater(w_high, w_low)

    def test_timestamp_alignment_prevents_leakage(self) -> None:
        # 2026-11-12 07:00 UTC = 15:00 HKT (before 16:00 close) -> same day 2026-11-12
        self.assertEqual(align_timestamp_to_trading_session("2026-11-12T07:00:00Z", "HKEX"), "2026-11-12")
        # 2026-11-12 09:00 UTC = 17:00 HKT (after 16:00 close) -> next trading day 2026-11-13
        self.assertEqual(align_timestamp_to_trading_session("2026-11-12T09:00:00Z", "HKEX"), "2026-11-13")
        # 2026-11-14 (Saturday) -> Monday 2026-11-16
        self.assertEqual(align_timestamp_to_trading_session("2026-11-14T10:00:00Z", "KRX"), "2026-11-16")

    def test_feature_engineering_and_walk_forward_validation(self) -> None:
        sample = json.loads((Path(__file__).resolve().parents[1] / "data" / "sample_dataset.json").read_text(encoding="utf-8"))
        bars = [b for b in sample["stock_prices"] if b["ticker"] == "0700.HK"]
        eng = FeatureEngineer()
        rows = eng.to_dicts(eng.build_features("0700.HK", "HKEX", bars, [], []))
        self.assertEqual(len(rows), len(bars))
        self.assertIsNone(rows[-1]["target_next_close"])
        self.assertEqual(rows[-2]["target_next_close"], rows[-1]["close"])

        evaluator = ModelEvaluator(eng.FEATURE_COLUMNS, min_train_size=12)
        results = evaluator.evaluate_all(rows)
        self.assertEqual(len(results), 4)  # Naive + Linear + RF + XGBoost
        self.assertEqual(sum(1 for r in results if r.is_best_model), 1)
        for r in results:
            self.assertGreaterEqual(r.directional_accuracy, 0.0)
            self.assertLessEqual(r.directional_accuracy, 100.0)

    def test_end_to_end_orchestrator_and_database_tables(self) -> None:
        from stock_sentiment_agent.collectors import PriceCollector
        from stock_sentiment_agent.preprocessing import FeatureEngineer

        orch = PipelineOrchestrator(db=self.db, use_hf=False)
        stock = StockConfig("0700.HK", "Tencent", "HKEX", "Communication Services")
        self.db.seed_stocks([stock])
        summary = orch.run_single(stock)

        self.assertIn("intelligence", summary)
        self.assertIn("prediction", summary)
        self.assertGreater(summary["intelligence"]["total_social_mentions"], 0)
        self.assertTrue(summary["data_provenance"]["contains_simulated_data"])

        # Priority 4: when live social/news are unavailable, ML training MUST exclude simulated social/news columns
        self.assertEqual(summary["data_provenance"]["genuine_posts_count"], 0)
        self.assertEqual(summary["data_provenance"]["genuine_news_count"], 0)
        self.assertEqual(summary["data_provenance"]["features_used"], FeatureEngineer.PRICE_FEATURE_COLUMNS)
        self.assertFalse(summary["data_provenance"]["is_demo_model"])

        for table in ("stocks", "social_posts", "news_articles", "stock_prices", "sentiment_results", "financial_features", "predictions", "model_metrics", "agent_logs"):
            rows = self.db.query_rows(f"SELECT COUNT(*) AS cnt FROM {table}")
            self.assertGreater(rows[0]["cnt"], 0, f"Expected rows in SQL table {table}")

        cats = summary["data_provenance"]["categories"]
        self.assertEqual(cats["social"], "SIMULATED DEMO DATA")
        self.assertEqual(cats["news"], "SIMULATED DEMO DATA")
        self.assertEqual(cats["prices"], "HISTORICAL DATA")
        self.assertIn("artifact_path", summary["prediction"])

        # Priority 3: verify PriceCollector refuses silent simulated fallback when allow_simulated_fallback=False
        pc = PriceCollector()
        with self.assertRaises(RuntimeError):
            pc.collect(
                StockConfig("INVALID_NO_PROVIDER.XX", "Invalid Test", "HKEX", "Tech"),
                allow_simulated_fallback=False,
                use_cache=False,
            )

        from stock_sentiment_agent.config import get_settings
        cfg_val = get_settings(reload=True).validate_configuration()
        self.assertIn("APIFY_API_TOKEN", cfg_val["checks"])
        self.assertIn("EXA_API_KEY", cfg_val["checks"])
        self.assertIn("DATABASE_URL", cfg_val["checks"])
        self.assertIn("MODEL_PATH", cfg_val["checks"])

    def test_assessment_report_generation(self) -> None:
        orch = PipelineOrchestrator(db=self.db, use_hf=False)
        tmp_path = Path(self.tmp_dir.name)
        orch.settings.data_dir = tmp_path
        orch.settings.reports_dir = tmp_path / "reports"
        orch.settings.model_path = tmp_path / "models"
        orch.reporting_agent.settings = orch.settings
        orch.prediction_agent.settings = orch.settings

        stock = StockConfig("005930.KS", "Samsung Electronics", "KRX", "Information Technology")
        result = orch.run_all([stock])
        self.assertIn("assessment_summary", result)
        summary = result["assessment_summary"]
        self.assertEqual(summary["stocks_analyzed"], 1)
        self.assertEqual(summary["genuine_posts_collected"], 0)
        self.assertEqual(summary["genuine_news_collected"], 0)
        self.assertEqual(len(summary["social_sources"]), 4)
        self.assertTrue(Path(result["report"]["assessment_report_path"]).exists())


if __name__ == "__main__":
    unittest.main()
