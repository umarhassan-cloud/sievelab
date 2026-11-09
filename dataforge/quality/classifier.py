"""Student educational-quality model: TF-IDF (char + word n-grams) -> ridge regression.

FineWeb-Edu uses a regression head over frozen embeddings, predicts a float in
[0, 5], and keeps documents with ``round(score) >= 3``. We mirror the
regression-then-threshold design with a model small enough to train in
seconds on CPU. Character n-grams (``char_wb``) make it script-agnostic, so a
single model serves English, Urdu and Roman Urdu.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import FeatureUnion, Pipeline


class EduQualityClassifier:
    def __init__(self, alpha: float = 1.0, max_features: int = 60000):
        self.model = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=max_features,
                                                     sublinear_tf=True, min_df=2)),
                            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=max_features // 2,
                                                     sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b\w+\b")),
                        ]
                    ),
                ),
                ("reg", Ridge(alpha=alpha)),
            ]
        )
        self.fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[float]) -> "EduQualityClassifier":
        self.model.fit(list(texts), np.asarray(labels, dtype=float))
        self.fitted = True
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.clip(self.model.predict(list(texts)), 0.0, 5.0)

    def predict_int(self, texts: Sequence[str]) -> np.ndarray:
        return np.rint(self.predict(texts)).astype(int)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "EduQualityClassifier":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def regression_report(y_true: Sequence[float], y_pred: Sequence[float], threshold: int = 3) -> Dict[str, float]:
    """FineWeb-Edu style metrics: MSE/MAE on the float score, quadratic-weighted
    Cohen's kappa on rounded scores, and binary F1 at the keep threshold."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    yt_int = np.rint(y_true).astype(int)
    yp_int = np.clip(np.rint(y_pred), 0, 5).astype(int)
    return {
        "mse": round(float(mean_squared_error(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "qwk": round(float(cohen_kappa_score(yt_int, yp_int, weights="quadratic")), 4),
        "exact_acc": round(float(np.mean(yt_int == yp_int)), 4),
        f"f1_at_{threshold}": round(float(f1_score(yt_int >= threshold, yp_int >= threshold)), 4),
        "n": int(len(y_true)),
    }
