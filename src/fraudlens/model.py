"""Models and the time-based split."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df.Amount)
    df["hour"] = (df.Time // 3600) % 24   # Time = seconds since the first transaction
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.6, valid_frac: float = 0.2):
    """Chronological split: train on the past, tune on the next slice, test on the latest.

    Random splits leak future fraud patterns into training and overstate performance.
    """
    df = df.sort_values("Time")
    a, b = df.Time.quantile(train_frac), df.Time.quantile(train_frac + valid_frac)
    return df[df.Time <= a], df[(df.Time > a) & (df.Time <= b)], df[df.Time > b]


def isolation_forest(seed=42):
    """Unsupervised anomaly detector: uses no fraud labels at all."""
    return IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1)


def logistic(seed=42):
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000,
                                                              C=0.1, random_state=seed))


def boosting(seed=42):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=None,
                                          max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, class_weight={0: 1, 1: 10},
                                          random_state=seed)
