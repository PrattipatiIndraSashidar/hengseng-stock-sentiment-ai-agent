"""Linear / Ridge Regression predictor for next-day closing price (n+1)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence


class LinearRegressionPredictor:
    """Predicts next-day close using scikit-learn Ridge or pure-Python normal equations."""

    name = "Linear Regression"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.feature_names: List[str] = []
        self._sk_model = None
        self._means: List[float] = []
        self._stds: List[float] = []
        self._weights: List[float] = []
        self._bias: float = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float], feature_names: List[str]) -> None:
        self.feature_names = list(feature_names)
        if not X or not y:
            raise ValueError("Training matrix X and target y must be non-empty.")

        try:
            from sklearn.linear_model import Ridge  # type: ignore
            from sklearn.pipeline import make_pipeline  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore

            pipe = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
            pipe.fit(X, y)
            self._sk_model = pipe
            return
        except ImportError:
            self._sk_model = None

        n_samples = len(X)
        n_features = len(X[0])
        self._means = [sum(row[j] for row in X) / n_samples for j in range(n_features)]
        self._stds = []
        for j in range(n_features):
            var = sum((row[j] - self._means[j]) ** 2 for row in X) / max(1, n_samples)
            std = var ** 0.5
            self._stds.append(std if std > 1e-8 else 1.0)

        Xs = [[(row[j] - self._means[j]) / self._stds[j] for j in range(n_features)] for row in X]
        self._bias = sum(y) / n_samples
        yc = [yi - self._bias for yi in y]

        # Normal equations: (X^T X + alpha * I) w = X^T y
        A = [[0.0] * n_features for _ in range(n_features)]
        b = [0.0] * n_features
        for i in range(n_samples):
            for j in range(n_features):
                b[j] += Xs[i][j] * yc[i]
                for k in range(n_features):
                    A[j][k] += Xs[i][j] * Xs[i][k]
        for j in range(n_features):
            A[j][j] += self.alpha

        self._weights = _solve_linear_system(A, b)

    def predict(self, X: Sequence[Sequence[float]]) -> List[float]:
        if self._sk_model is not None:
            return [float(v) for v in self._sk_model.predict(X)]
        preds: List[float] = []
        for row in X:
            z = [(row[j] - self._means[j]) / self._stds[j] for j in range(len(self._weights))]
            preds.append(self._bias + sum(w * x for w, x in zip(self._weights, z)))
        return preds

    def feature_importances(self) -> Dict[str, float]:
        if self._sk_model is not None:
            coefs = self._sk_model.named_steps["ridge"].coef_
            raw = [abs(float(c)) for c in coefs]
        else:
            raw = [abs(float(w)) for w in self._weights]
        total = sum(raw) or 1.0
        return {name: round(val / total, 4) for name, val in zip(self.feature_names, raw)}

    def save(self, path: Path) -> None:
        try:
            import joblib  # type: ignore

            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
        except ImportError:
            pass


def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        denom = M[col][col] if abs(M[col][col]) > 1e-12 else 1e-12
        for j in range(col, n + 1):
            M[col][j] /= denom
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]
