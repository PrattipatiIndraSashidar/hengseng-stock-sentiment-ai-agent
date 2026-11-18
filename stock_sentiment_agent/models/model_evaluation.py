"""Walk-forward chronological validation, naive baseline comparison, and conformal uncertainty intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from stock_sentiment_agent.models.linear_model import LinearRegressionPredictor
from stock_sentiment_agent.models.random_forest import RandomForestPredictor
from stock_sentiment_agent.models.xgboost_model import XGBoostPredictor


def _empirical_conformal_quantile(abs_pct_errors: Sequence[float], alpha: float) -> float:
    """Finite-sample split conformal quantile over absolute percentage residuals."""
    n = len(abs_pct_errors)
    if n == 0:
        return 0.02 if alpha >= 0.15 else 0.035
    ordered = sorted(abs(float(e)) for e in abs_pct_errors)
    rank = min(n, max(1, math.ceil((n + 1) * (1.0 - alpha)))) - 1
    return float(ordered[rank])


def compute_metrics(
    actuals: Sequence[float],
    predictions: Sequence[float],
    prev_closes: Sequence[float],
) -> Dict[str, float]:
    """Calculate MAE, RMSE, MAPE, Directional Accuracy, and residual dispersion."""
    n = len(actuals)
    if n == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "directional_accuracy": 0.0,
            "residual_mean": 0.0,
            "residual_std": 0.0,
        }

    residuals = [p - a for a, p in zip(actuals, predictions)]
    abs_errors = [abs(r) for r in residuals]
    sq_errors = [r ** 2 for r in residuals]
    pct_errors = [abs(a - p) / max(abs(a), 1e-8) * 100.0 for a, p in zip(actuals, predictions)]

    mean_res = sum(residuals) / n
    var_res = sum((r - mean_res) ** 2 for r in residuals) / max(1, n - 1)
    std_res = math.sqrt(var_res)

    dir_hits = 0
    for a, p, prev in zip(actuals, predictions, prev_closes):
        actual_dir = 1 if a >= prev else -1
        pred_dir = 1 if p >= prev else -1
        if actual_dir == pred_dir:
            dir_hits += 1

    return {
        "mae": round(sum(abs_errors) / n, 4),
        "rmse": round(math.sqrt(sum(sq_errors) / n), 4),
        "mape": round(sum(pct_errors) / n, 4),
        "directional_accuracy": round((dir_hits / n) * 100.0, 2),
        "residual_mean": round(mean_res, 4),
        "residual_std": round(std_res, 4),
    }


@dataclass
class WalkForwardResult:
    model_name: str
    split_window: str
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    predictions: List[Dict[str, Any]]
    feature_importances: Dict[str, float]
    is_best_model: bool = False
    residual_mean: float = 0.0
    residual_std: float = 0.0
    conformal_pct_80: float = 0.0
    conformal_pct_95: float = 0.0
    empirical_coverage_80: float = 0.0
    rmse_vs_naive_pct: float = 0.0
    mae_vs_naive_pct: float = 0.0
    dir_acc_vs_naive_pp: float = 0.0
    beats_naive_rmse: bool = False


class ModelEvaluator:
    """Performs strictly chronological walk-forward validation predicting next-day returns (no leakage)."""

    def __init__(self, feature_columns: List[str], min_train_size: int = 12) -> None:
        self.feature_columns = feature_columns
        self.min_train_size = min_train_size

    def evaluate_all(
        self,
        feature_rows: List[Dict[str, Any]],
    ) -> List[WalkForwardResult]:
        labeled = sorted(
            [r for r in feature_rows if r.get("target_next_close") is not None],
            key=lambda r: r["trading_date"],
        )
        if len(labeled) < self.min_train_size + 2:
            raise ValueError(
                f"Need at least {self.min_train_size + 2} labeled sessions for walk-forward validation; got {len(labeled)}."
            )

        split_idx = max(self.min_train_size, int(len(labeled) * 0.65))
        split_window = f"{labeled[split_idx]['trading_date']} -> {labeled[-1]['trading_date']}"

        model_factories = [
            lambda: LinearRegressionPredictor(alpha=30.0),
            lambda: RandomForestPredictor(n_estimators=60, max_depth=3),
            lambda: XGBoostPredictor(n_estimators=60, learning_rate=0.04, max_depth=2),
        ]

        results: List[WalkForwardResult] = []

        # 1. Naive Previous-Close Baseline (Close[t+1] = Close[t], 0% expected return)
        naive_actuals, naive_preds, naive_prevs, naive_records = [], [], [], []
        naive_pct_res: List[float] = []
        for idx in range(split_idx, len(labeled)):
            row = labeled[idx]
            prev_c = float(row["close"])
            actual_c = float(row["target_next_close"])
            naive_actuals.append(actual_c)
            naive_preds.append(prev_c)
            naive_prevs.append(prev_c)
            pct_err = abs(actual_c - prev_c) / max(prev_c, 1e-8)
            q80 = _empirical_conformal_quantile(naive_pct_res, 0.20) if len(naive_pct_res) >= 4 else 0.025
            naive_pct_res.append(pct_err)
            naive_records.append(
                {
                    "prediction_date": row["trading_date"],
                    "last_close": prev_c,
                    "predicted_close": round(prev_c, 4),
                    "actual_close": round(actual_c, 4),
                    "ci_80_low": round(prev_c * (1.0 - q80), 4),
                    "ci_80_high": round(prev_c * (1.0 + q80), 4),
                    "error": round(prev_c - actual_c, 4),
                    "dir_hit": (actual_c >= prev_c),
                }
            )
        naive_m = compute_metrics(naive_actuals, naive_preds, naive_prevs)
        naive_q80 = _empirical_conformal_quantile(naive_pct_res, 0.20)
        naive_q95 = _empirical_conformal_quantile(naive_pct_res, 0.05)
        naive_cov80 = (
            sum(1 for rec in naive_records if rec["ci_80_low"] <= rec["actual_close"] <= rec["ci_80_high"])
            / max(1, len(naive_records))
            * 100.0
        )
        results.append(
            WalkForwardResult(
                model_name="Naive Baseline (Prev Close)",
                split_window=split_window,
                mae=naive_m["mae"],
                rmse=naive_m["rmse"],
                mape=naive_m["mape"],
                directional_accuracy=naive_m["directional_accuracy"],
                predictions=naive_records,
                feature_importances={},
                is_best_model=False,
                residual_mean=naive_m["residual_mean"],
                residual_std=naive_m["residual_std"],
                conformal_pct_80=round(naive_q80 * 100.0, 2),
                conformal_pct_95=round(naive_q95 * 100.0, 2),
                empirical_coverage_80=round(naive_cov80, 1),
                rmse_vs_naive_pct=0.0,
                mae_vs_naive_pct=0.0,
                dir_acc_vs_naive_pp=0.0,
                beats_naive_rmse=True,
            )
        )

        # 2. Walk-Forward Expanding Window predicting stationary return r_{t+1}
        for factory in model_factories:
            actuals: List[float] = []
            preds: List[float] = []
            prevs: List[float] = []
            records: List[Dict[str, Any]] = []
            causal_pct_res: List[float] = []
            last_model = None

            for idx in range(split_idx, len(labeled)):
                train_slice = labeled[:idx]
                test_row = labeled[idx]
                X_train = [[float(r[c]) for c in self.feature_columns] for r in train_slice]
                y_ret_train = [
                    (float(r["target_next_close"]) - float(r["close"])) / max(float(r["close"]), 1e-8)
                    for r in train_slice
                ]
                X_test = [[float(test_row[c]) for c in self.feature_columns]]

                model = factory()
                model.fit(X_train, y_ret_train, self.feature_columns)
                pred_ret = float(model.predict(X_test)[0])
                pred_ret = max(-0.15, min(0.15, pred_ret))

                prev_close = float(test_row["close"])
                pred_val = prev_close * (1.0 + pred_ret)
                actual_val = float(test_row["target_next_close"])

                # Strictly causal split-conformal 80% interval from prior walk-forward residuals
                q80_step = _empirical_conformal_quantile(causal_pct_res, 0.20) if len(causal_pct_res) >= 4 else 0.025
                pct_err = abs(actual_val - pred_val) / max(pred_val, 1e-8)
                causal_pct_res.append(pct_err)

                actuals.append(actual_val)
                preds.append(pred_val)
                prevs.append(prev_close)
                records.append(
                    {
                        "prediction_date": test_row["trading_date"],
                        "last_close": prev_close,
                        "predicted_close": round(pred_val, 4),
                        "actual_close": round(actual_val, 4),
                        "ci_80_low": round(pred_val * (1.0 - q80_step), 4),
                        "ci_80_high": round(pred_val * (1.0 + q80_step), 4),
                        "error": round(pred_val - actual_val, 4),
                        "dir_hit": ((actual_val >= prev_close) == (pred_val >= prev_close)),
                    }
                )
                last_model = model

            metrics = compute_metrics(actuals, preds, prevs)
            q80_final = _empirical_conformal_quantile(causal_pct_res, 0.20)
            q95_final = _empirical_conformal_quantile(causal_pct_res, 0.05)
            cov80 = (
                sum(1 for rec in records if rec["ci_80_low"] <= rec["actual_close"] <= rec["ci_80_high"])
                / max(1, len(records))
                * 100.0
            )
            rmse_diff_pct = ((metrics["rmse"] - naive_m["rmse"]) / max(naive_m["rmse"], 1e-8)) * 100.0
            mae_diff_pct = ((metrics["mae"] - naive_m["mae"]) / max(naive_m["mae"], 1e-8)) * 100.0
            dir_diff_pp = metrics["directional_accuracy"] - naive_m["directional_accuracy"]

            results.append(
                WalkForwardResult(
                    model_name=last_model.name if last_model else "Unknown",
                    split_window=split_window,
                    mae=metrics["mae"],
                    rmse=metrics["rmse"],
                    mape=metrics["mape"],
                    directional_accuracy=metrics["directional_accuracy"],
                    predictions=records,
                    feature_importances=last_model.feature_importances() if last_model else {},
                    is_best_model=False,
                    residual_mean=metrics["residual_mean"],
                    residual_std=metrics["residual_std"],
                    conformal_pct_80=round(q80_final * 100.0, 2),
                    conformal_pct_95=round(q95_final * 100.0, 2),
                    empirical_coverage_80=round(cov80, 1),
                    rmse_vs_naive_pct=round(rmse_diff_pct, 2),
                    mae_vs_naive_pct=round(mae_diff_pct, 2),
                    dir_acc_vs_naive_pp=round(dir_diff_pp, 2),
                    beats_naive_rmse=metrics["rmse"] <= naive_m["rmse"],
                )
            )

        # Select best ML model strictly by lowest out-of-sample RMSE (tie-break MAE)
        ml_only = [r for r in results if not r.model_name.startswith("Naive")]
        if ml_only:
            best = min(ml_only, key=lambda r: (r.rmse, r.mae))
            best.is_best_model = True

        return results
