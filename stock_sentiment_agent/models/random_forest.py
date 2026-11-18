"""Random Forest Regressor for next-day closing price (n+1) with chronological tuning."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


class _SimpleTree:
    """Lightweight CART regression tree for stdlib-only environments."""

    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree: Optional[Dict] = None
        self.feature_gains: Dict[int, float] = {}

    def fit(self, X: List[List[float]], y: List[float], feature_subset: List[int]) -> None:
        self.feature_gains = {}
        self.tree = self._build(X, y, feature_subset, depth=0)

    def _build(self, X: List[List[float]], y: List[float], feats: List[int], depth: int) -> Dict:
        mean_y = sum(y) / len(y)
        if depth >= self.max_depth or len(y) <= 2 * self.min_samples_leaf:
            return {"leaf": True, "value": mean_y}

        base_sse = sum((v - mean_y) ** 2 for v in y)
        best_gain = 0.0
        best_feat = None
        best_thresh = None
        best_split = None

        for f in feats:
            vals = sorted(set(row[f] for row in X))
            if len(vals) <= 1:
                continue
            step = max(1, (len(vals) - 1) // 8)
            for idx in range(0, len(vals) - 1, step):
                thresh = 0.5 * (vals[idx] + vals[idx + 1])
                left_idx = [i for i, r in enumerate(X) if r[f] <= thresh]
                right_idx = [i for i, r in enumerate(X) if r[f] > thresh]
                if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                    continue
                ly = [y[i] for i in left_idx]
                ry = [y[i] for i in right_idx]
                lm, rm = sum(ly) / len(ly), sum(ry) / len(ry)
                sse = sum((v - lm) ** 2 for v in ly) + sum((v - rm) ** 2 for v in ry)
                gain = base_sse - sse
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh
                    best_split = (left_idx, right_idx)

        if best_feat is None or best_split is None:
            return {"leaf": True, "value": mean_y}

        self.feature_gains[best_feat] = self.feature_gains.get(best_feat, 0.0) + best_gain
        li, ri = best_split
        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build([X[i] for i in li], [y[i] for i in li], feats, depth + 1),
            "right": self._build([X[i] for i in ri], [y[i] for i in ri], feats, depth + 1),
        }

    def predict_one(self, row: Sequence[float]) -> float:
        node = self.tree
        while node and not node["leaf"]:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return float(node["value"]) if node else 0.0


class RandomForestPredictor:
    """Predicts next-day close using sklearn RandomForestRegressor or pure-Python bagged trees."""

    name = "Random Forest Regressor"

    def __init__(self, n_estimators: int = 80, max_depth: int = 5, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.feature_names: List[str] = []
        self._sk_model = None
        self._trees: List[_SimpleTree] = []
        self._importances: Dict[str, float] = {}

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float], feature_names: List[str]) -> None:
        self.feature_names = list(feature_names)
        if not X or not y:
            raise ValueError("Training data cannot be empty.")

        try:
            from sklearn.ensemble import RandomForestRegressor  # type: ignore
            from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # type: ignore

            if len(X) >= 15:
                tscv = TimeSeriesSplit(n_splits=3)
                grid = GridSearchCV(
                    RandomForestRegressor(random_state=self.random_state),
                    param_grid={"n_estimators": [60, 120], "max_depth": [4, 6]},
                    cv=tscv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=1,
                )
                grid.fit(X, y)
                self._sk_model = grid.best_estimator_
            else:
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state
                )
                model.fit(X, y)
                self._sk_model = model
            return
        except ImportError:
            self._sk_model = None

        rng = random.Random(self.random_state)
        n_samples = len(X)
        n_features = len(X[0])
        m_try = max(2, int(n_features ** 0.5))
        n_trees = min(12, self.n_estimators)
        self._trees = []
        total_gains: Dict[int, float] = {}

        X_list = [list(r) for r in X]
        y_list = [float(v) for v in y]

        for _ in range(n_trees):
            indices = [rng.randrange(n_samples) for _ in range(n_samples)]
            Xb = [X_list[i] for i in indices]
            yb = [y_list[i] for i in indices]
            feats = rng.sample(range(n_features), min(m_try, n_features))
            tree = _SimpleTree(max_depth=min(3, self.max_depth), min_samples_leaf=2)
            tree.fit(Xb, yb, feats)
            self._trees.append(tree)
            for f_idx, g in tree.feature_gains.items():
                total_gains[f_idx] = total_gains.get(f_idx, 0.0) + g

        norm = sum(total_gains.values()) or 1.0
        self._importances = {
            self.feature_names[i]: round(total_gains.get(i, 0.0) / norm, 4) for i in range(n_features)
        }

    def predict(self, X: Sequence[Sequence[float]]) -> List[float]:
        if self._sk_model is not None:
            return [float(v) for v in self._sk_model.predict(X)]
        return [sum(t.predict_one(row) for t in self._trees) / len(self._trees) for row in X]

    def feature_importances(self) -> Dict[str, float]:
        if self._sk_model is not None:
            imps = self._sk_model.feature_importances_
            return {n: round(float(v), 4) for n, v in zip(self.feature_names, imps)}
        return self._importances

    def save(self, path: Path) -> None:
        try:
            import joblib  # type: ignore

            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
        except ImportError:
            pass
