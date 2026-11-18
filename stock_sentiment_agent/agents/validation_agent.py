"""Agent 7: Model Validation Agent (walk-forward out-of-sample evaluation & realized reconciliation)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from stock_sentiment_agent.config import StockConfig
from stock_sentiment_agent.database import DatabaseManager
from stock_sentiment_agent.models import ModelEvaluator, WalkForwardResult


class ModelValidationAgent:
    AGENT_NAME = "ModelValidationAgent"

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def run(
        self,
        stock: StockConfig,
        feature_rows: List[Dict[str, Any]],
        feature_columns: List[str],
    ) -> List[WalkForwardResult]:
        started = time.perf_counter()
        evaluator = ModelEvaluator(feature_columns=feature_columns, min_train_size=12)
        results = evaluator.evaluate_all(feature_rows)
        self._persist_metrics(stock.ticker, results)
        self._reconcile_past_predictions(stock.ticker, feature_rows)

        best = next((r for r in results if r.is_best_model), results[-1])
        naive = next((r for r in results if r.model_name.startswith("Naive")), None)
        vs_naive = f", vsNaive={best.rmse_vs_naive_pct:+.1f}%" if naive else ""
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self.db.log_agent(
            agent_name=self.AGENT_NAME,
            status="SUCCESS",
            ticker=stock.ticker,
            message=f"Validated {len(results)} models; best={best.model_name} (RMSE={best.rmse}{vs_naive}, DirAcc={best.directional_accuracy}%).",
            records_processed=len(results),
            duration_ms=elapsed_ms,
        )
        return results

    def _persist_metrics(self, ticker: str, results: List[WalkForwardResult]) -> None:
        if self.db.sqlite_path is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self.db.sqlite_connection() as conn:
            conn.execute("DELETE FROM model_metrics WHERE ticker=?", (ticker,))
            for r in results:
                conn.execute(
                    """
                    INSERT INTO model_metrics (
                        ticker, model_name, split_window, mae, rmse, mape,
                        directional_accuracy, is_best_model, feature_importances_json, evaluated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker, r.model_name, r.split_window, r.mae, r.rmse, r.mape,
                        r.directional_accuracy, int(r.is_best_model),
                        json.dumps(r.feature_importances), now,
                    ),
                )

    def _reconcile_past_predictions(self, ticker: str, feature_rows: List[Dict[str, Any]]) -> None:
        """Compare stored predictions against actual closing prices once the target session finishes."""
        if self.db.sqlite_path is None:
            return
        close_by_date = {r["trading_date"]: float(r["close"]) for r in feature_rows}
        with self.db.sqlite_connection() as conn:
            rows = conn.execute(
                "SELECT id, forecast_target_date, last_close, predicted_close FROM predictions WHERE ticker=? AND is_validated=0",
                (ticker,),
            ).fetchall()
            for row in rows:
                target = row["forecast_target_date"]
                if target in close_by_date:
                    actual = close_by_date[target]
                    pred = float(row["predicted_close"])
                    last_c = float(row["last_close"])
                    mae = abs(actual - pred)
                    mape = mae / max(actual, 1e-8) * 100.0
                    dir_hit = 100.0 if ((actual >= last_c) == (pred >= last_c)) else 0.0
                    conn.execute(
                        """
                        UPDATE predictions
                        SET actual_close=?, mae=?, rmse=?, mape=?, directional_accuracy=?, is_validated=1
                        WHERE id=?
                        """,
                        (actual, round(mae, 4), round(mae, 4), round(mape, 4), dir_hit, row["id"]),
                    )
