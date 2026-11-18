"""Agent 6: Price Prediction Agent (n+1 return forecast + conformal uncertainty intervals)."""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List

from stock_sentiment_agent.config import StockConfig, get_settings
from stock_sentiment_agent.database import DatabaseManager
from stock_sentiment_agent.models import (
    LinearRegressionPredictor,
    RandomForestPredictor,
    WalkForwardResult,
    XGBoostPredictor,
)
from stock_sentiment_agent.preprocessing import next_trading_day


class PricePredictionAgent:
    AGENT_NAME = "PricePredictionAgent"

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self.settings = get_settings()

    def run(
        self,
        stock: StockConfig,
        feature_rows: List[Dict[str, Any]],
        feature_columns: List[str],
        validation_results: List[WalkForwardResult],
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        labeled = [r for r in feature_rows if r.get("target_next_close") is not None]
        if len(labeled) < 8:
            raise ValueError(f"Insufficient labeled history to train prediction model for {stock.ticker}")

        best_val = next((r for r in validation_results if r.is_best_model), None)
        naive_val = next((r for r in validation_results if r.model_name.startswith("Naive")), None)
        best_model_name = best_val.model_name if best_val else "Random Forest Regressor"

        factory_map = {
            "Linear Regression": lambda: LinearRegressionPredictor(alpha=30.0),
            "Random Forest Regressor": lambda: RandomForestPredictor(n_estimators=60, max_depth=3),
            "XGBoost Regressor": lambda: XGBoostPredictor(n_estimators=60, learning_rate=0.04, max_depth=2),
        }
        model = factory_map.get(best_model_name, factory_map["Random Forest Regressor"])()

        X_train = [[float(r[c]) for c in feature_columns] for r in labeled]
        y_ret_train = [
            (float(r["target_next_close"]) - float(r["close"])) / max(float(r["close"]), 1e-8)
            for r in labeled
        ]
        model.fit(X_train, y_ret_train, feature_columns)

        latest_row = feature_rows[-1]
        X_next = [[float(latest_row[c]) for c in feature_columns]]
        pred_ret = float(model.predict(X_next)[0])
        pred_ret = max(-0.15, min(0.15, pred_ret))

        last_close = float(latest_row["close"])
        predicted_close = last_close * (1.0 + pred_ret)
        pct_change = pred_ret * 100.0

        # Conformal 80% & 95% uncertainty intervals from historical out-of-sample residuals
        q80 = (best_val.conformal_pct_80 / 100.0) if (best_val and best_val.conformal_pct_80 > 0) else 0.025
        q95 = (best_val.conformal_pct_95 / 100.0) if (best_val and best_val.conformal_pct_95 > 0) else 0.045
        ci_80_low = predicted_close * (1.0 - q80)
        ci_80_high = predicted_close * (1.0 + q80)
        ci_95_low = predicted_close * (1.0 - q95)
        ci_95_high = predicted_close * (1.0 + q95)

        pred_date = latest_row["trading_date"]
        target_date = next_trading_day(date.fromisoformat(pred_date), stock.exchange).isoformat()

        slug = f"{stock.ticker.replace('.', '_')}_{best_model_name.lower().replace(' ', '_')}"
        self.settings.model_path.mkdir(parents=True, exist_ok=True)
        model_file = self.settings.model_path / f"{slug}.joblib"
        model.save(model_file)
        meta_file = self.settings.model_path / f"{slug}.json"
        meta_file.write_text(
            json.dumps(
                {
                    "ticker": stock.ticker,
                    "model_name": best_model_name,
                    "trained_samples": len(labeled),
                    "feature_columns": feature_columns,
                    "feature_importances": model.feature_importances(),
                    "prediction_date": pred_date,
                    "forecast_target_date": target_date,
                    "ci_80": [round(ci_80_low, 4), round(ci_80_high, 4)],
                    "ci_95": [round(ci_95_low, 4), round(ci_95_high, 4)],
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        prediction_record = {
            "ticker": stock.ticker,
            "prediction_date": pred_date,
            "forecast_target_date": target_date,
            "last_close": round(last_close, 4),
            "predicted_close": round(predicted_close, 4),
            "actual_close": None,
            "predicted_pct_change": round(pct_change, 2),
            "model_name": best_model_name,
            "artifact_path": str(meta_file.name),
            "mae": best_val.mae if best_val else None,
            "rmse": best_val.rmse if best_val else None,
            "mape": best_val.mape if best_val else None,
            "directional_accuracy": best_val.directional_accuracy if best_val else None,
            "naive_rmse": naive_val.rmse if naive_val else None,
            "rmse_vs_naive_pct": best_val.rmse_vs_naive_pct if best_val else None,
            "beats_naive_rmse": best_val.beats_naive_rmse if best_val else None,
            "ci_80_low": round(ci_80_low, 4),
            "ci_80_high": round(ci_80_high, 4),
            "ci_95_low": round(ci_95_low, 4),
            "ci_95_high": round(ci_95_high, 4),
            "empirical_coverage_80": best_val.empirical_coverage_80 if best_val else None,
            "is_validated": False,
        }
        self._persist_prediction(prediction_record)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self.db.log_agent(
            agent_name=self.AGENT_NAME,
            status="SUCCESS",
            ticker=stock.ticker,
            message=f"Forecast {target_date}: {predicted_close:.2f} ({pct_change:+.2f}%) [80% CI: {ci_80_low:.2f}..{ci_80_high:.2f}] via {best_model_name}.",
            records_processed=1,
            duration_ms=elapsed_ms,
        )
        return prediction_record

    def _persist_prediction(self, pred: Dict[str, Any]) -> None:
        if self.db.sqlite_path is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self.db.sqlite_connection() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    ticker, prediction_date, forecast_target_date, last_close,
                    predicted_close, actual_close, predicted_pct_change, model_name,
                    mae, rmse, mape, directional_accuracy, is_validated, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pred["ticker"], pred["prediction_date"], pred["forecast_target_date"],
                    pred["last_close"], pred["predicted_close"], pred["actual_close"],
                    pred["predicted_pct_change"], pred["model_name"], pred["mae"],
                    pred["rmse"], pred["mape"], pred["directional_accuracy"],
                    int(pred["is_validated"]), now,
                ),
            )
