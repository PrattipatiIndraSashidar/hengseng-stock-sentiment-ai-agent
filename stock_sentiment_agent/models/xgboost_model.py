"""XGBoost Regressor for next-day closing price (n+1) with chronological hyperparameter tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from stock_sentiment_agent.models.random_forest import _SimpleTree


class XGBoostPredictor:
    """Predicts next-day close using xgboost.XGBRegressor or gradient-boosted trees fallback."""

    name = "XGBoost Regressor"

    def __init__(self, n_estimators: int = 80, learning_rate: float = 0.08, max_depth: int = 3, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.feature_names: List[str] = []
        self._xgb_model = None
        self._base_pred: float = 0.0
        self._trees: List[_SimpleTree] = []
        self._importances: Dict[str, float] = {}

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float], feature_names: List[str]) -> None:
        self.feature_names = list(feature_names)
        if not X or not y:
            raise ValueError("Training data cannot be empty.")

        try:
            from xgboost import XGBRegressor  # type: ignore
            from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # type: ignore

            if len(X) >= 15:
                tscv = TimeSeriesSplit(n_splits=3)
                grid = GridSearchCV(
                    XGBRegressor(random_state=self.random_state, objective="reg:squarederror", verbosity=0),
                    param_grid={"n_estimators": [50, 100], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
                    cv=tscv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=1,
                )
                grid.fit(X, y)
                self._xgb_model = grid.best_estimator_
            else:
                model = XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                    objective="reg:squarederror",
                    verbosity=0,
                )
                model.fit(X, y)
                self._xgb_model = model
            return
        except ImportError:
            self._xgb_model = None

        # Pure-Python Gradient Boosted Regression Trees fallback
        X_list = [list(r) for r in X]
        y_list = [float(v) for v in y]
        n_features = len(X_list[0])
        self._base_pred = sum(y_list) / len(y_list)
        preds = [self._base_pred] * len(y_list)
        self._trees = []
        total_gains: Dict[int, float] = {}

        steps = min(8, self.n_estimators)
        all_feats = list(range(n_features))
        for s_idx in range(steps):
            residuals = [yt - yp for yt, yp in zip(y_list, preds)]
            sub_feats = all_feats[s_idx % 3 :: 3] or all_feats
            tree = _SimpleTree(max_depth=min(2, self.max_depth), min_samples_leaf=2)
            tree.fit(X_list, residuals, sub_feats)
            self._trees.append(tree)
            for idx, row in enumerate(X_list):
                preds[idx] += self.learning_rate * tree.predict_one(row)
            for f_idx, g in tree.feature_gains.items():
                total_gains[f_idx] = total_gains.get(f_idx, 0.0) + g

        norm = sum(total_gains.values()) or 1.0
        self._importances = {
            self.feature_names[i]: round(total_gains.get(i, 0.0) / norm, 4) for i in range(n_features)
        }

    def predict(self, X: Sequence[Sequence[float]]) -> List[float]:
        if self._xgb_model is not None:
            return [float(v) for v in self._xgb_model.predict(X)]
        out: List[float] = []
        for row in X:
            val = self._base_pred + sum(self.learning_rate * t.predict_one(row) for t in self._trees)
            out.append(val)
        return out

    def feature_importances(self) -> Dict[str, float]:
        if self._xgb_model is not None:
            imps = self._xgb_model.feature_importances_
            return {n: round(float(v), 4) for n, v in zip(self.feature_names, imps)}
        return self._importances

    def save(self, path: Path) -> None:
        try:
            import joblib  # type: ignore

            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
        except ImportError:
            pass
