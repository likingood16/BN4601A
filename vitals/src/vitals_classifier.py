"""Features and model definitions for the vitals classifier.

Inputs are limited to what the app will have for a new patient: vitals,
consciousness, pain score, age, sex and whether it is an injury. The chief
complaint is deliberately left out (the main language model reads the story),
as are arrival mode, ED type and crowding, which are specific to the Korean sites.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import news2

VITALS = ["sbp", "dbp", "hr", "rr", "temp_c", "spo2"]
FEATURES = ["age", "is_male", "injured", "mental", "nrs_pain", *VITALS, "news2"]
URGENT_MAX_KTAS = 2      # "urgent" = KTAS 1-2


def build_X(df):
    """Feature matrix from a cleaned KTAS table (or app input in the same columns).
    Missing values stay NaN; each model handles them itself."""
    X = df.copy()
    if "news2" not in X.columns:
        X["news2"] = news2.score_frame(X)["news2"].to_numpy()
    return X[FEATURES].astype(float)


def urgent_target(df):
    return (df["ktas_expert"] <= URGENT_MAX_KTAS).astype(int)


def make_logreg(C=1.0):
    # add_indicator=True gives the model a "was missing" column per vital
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000))


def make_hgb(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, seed=0):
    # Handles NaN natively, so missingness itself can be learnt as a signal
    return HistGradientBoostingClassifier(max_depth=max_depth, learning_rate=learning_rate,
                                          max_iter=max_iter, min_samples_leaf=min_samples_leaf,
                                          l2_regularization=1.0, random_state=seed)


def urgent_proba(model, X):
    """P(urgent). Works for a binary model or a 5-class KTAS model (P(1) + P(2))."""
    p = model.predict_proba(X)
    classes = list(model.classes_)
    if set(classes) <= {0, 1}:
        return p[:, classes.index(1)]
    return p[:, [i for i, c in enumerate(classes) if c <= URGENT_MAX_KTAS]].sum(axis=1)


def bootstrap(metric, y, *scores, n_boot=2000, seed=0):
    """Point estimate and 95% CI of metric(y, s); with two scores, of their difference."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = [np.asarray(s) for s in scores]
    f = (lambda i: metric(y[i], scores[0][i])) if len(scores) == 1 else \
        (lambda i: metric(y[i], scores[0][i]) - metric(y[i], scores[1][i]))
    est = f(np.arange(len(y)))
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            stats.append(f(i))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return est, lo, hi
