"""Business-focused evaluation: what does a fraud model save, after paying for reviews?

Cost model (all in euros):
  * a fraud that is not flagged costs its full transaction amount (the bank refunds it)
  * every flagged transaction costs a fixed review cost (an analyst checks it)
  * a fraud that is flagged and reviewed is stopped, so it costs only the review
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

REVIEW_COST = 5.0   # euros per manual review, a deliberately simple assumption


def cost(y, amount, flagged, review_cost: float = REVIEW_COST) -> float:
    y = np.asarray(y).astype(bool); flagged = np.asarray(flagged).astype(bool)
    amount = np.asarray(amount, float)
    return float(amount[y & ~flagged].sum() + review_cost * flagged.sum())


def best_threshold(y, amount, score, review_cost: float = REVIEW_COST) -> float:
    """Threshold that minimises total cost on the data given (use validation data only)."""
    cands = np.unique(np.quantile(score, np.linspace(0.90, 0.99999, 400)))
    costs = [cost(y, amount, score >= c, review_cost) for c in cands]
    return float(cands[int(np.argmin(costs))])


def recall_at_budget(y, score, budget: float) -> float:
    """Share of frauds caught if analysts can review only the top `budget` share of transactions."""
    k = max(1, int(round(budget * len(score))))
    top = np.argsort(-np.asarray(score))[:k]
    return float(np.asarray(y)[top].sum() / np.asarray(y).sum())


def summarise(name, y, amount, score, threshold, review_cost: float = REVIEW_COST) -> dict:
    flagged = score >= threshold
    y = np.asarray(y).astype(bool)
    return {
        "model": name,
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "recall_top_0.1pct": recall_at_budget(y, score, 0.001),
        "recall_top_0.5pct": recall_at_budget(y, score, 0.005),
        "alerts": int(flagged.sum()),
        "frauds_caught": int((y & flagged).sum()),
        "frauds_total": int(y.sum()),
        "precision": float((y & flagged).sum() / max(1, flagged.sum())),
        "fraud_value_stopped": float(np.asarray(amount)[y & flagged].sum()),
        "total_cost": cost(y, amount, flagged, review_cost),
    }


def pr_curve(y, score):
    p, r, _ = precision_recall_curve(y, score)
    return pd.DataFrame({"precision": p, "recall": r})
