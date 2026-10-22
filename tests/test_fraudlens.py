import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudlens.data import validate
from fraudlens.evaluate import best_threshold, cost, recall_at_budget
from fraudlens.model import add_features, time_split


def test_cost_counts_missed_fraud_and_reviews():
    y = [1, 1, 0, 0]; amount = [100, 50, 10, 10]; flagged = [True, False, True, False]
    # missed fraud: 50; two reviews at 5 each: 10
    assert cost(y, amount, flagged, review_cost=5) == 60


def test_threshold_prefers_flagging_only_expensive_fraud():
    y = np.array([1, 0, 0, 0]); amount = np.array([500.0, 1, 1, 1]); score = np.array([0.9, 0.2, 0.1, 0.05])
    t = best_threshold(y, amount, score, review_cost=5)
    assert (score >= t).tolist() == [True, False, False, False]


def test_recall_at_budget():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0]); s = np.arange(10)[::-1] / 10
    assert recall_at_budget(y, s, 0.1) == 0.5
    assert recall_at_budget(y, s, 0.3) == 1.0


def test_time_split_is_chronological_and_disjoint():
    df = pd.DataFrame({"Time": np.arange(1000), "Class": 0})
    tr, va, te = time_split(df)
    assert tr.Time.max() < va.Time.min() and va.Time.max() < te.Time.min()
    assert len(tr) + len(va) + len(te) == 1000


def test_features_hour_and_log_amount():
    df = add_features(pd.DataFrame({"Time": [0, 3600 * 25], "Amount": [0.0, np.e - 1]}))
    assert df.hour.tolist() == [0, 1] and df.log_amount.iloc[1] == pytest.approx(1.0)


def test_validate_rejects_wrong_file():
    with pytest.raises(AssertionError):
        validate(pd.DataFrame({"Time": [0], "Amount": [1.0], "Class": [0]}))


DATA = ROOT / "data" / "raw" / "creditcard.csv"


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/run_analysis.py first")
def test_real_file_matches_published_counts():
    validate(pd.read_csv(DATA))
