"""Train, tune and evaluate fraud models on a time-based split; write results and figures."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudlens.data import load
from fraudlens.evaluate import REVIEW_COST, best_threshold, cost, pr_curve, summarise
from fraudlens.model import FEATURES, add_features, boosting, isolation_forest, logistic, time_split

FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
NAVY, TERRA, TEAL, GREY = "#1f3a5f", "#c8553d", "#2a9d8f", "#8a8f98"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})

df = add_features(load(ROOT / "data" / "raw" / "creditcard.csv"))
train, valid, test = time_split(df)
print({k: (len(v), int(v.Class.sum())) for k, v in [("train", train), ("valid", valid), ("test", test)]})

models, scores_valid, scores_test = {}, {}, {}
iso = isolation_forest().fit(train[FEATURES])           # trained without labels
scores_valid["Isolation forest (no labels)"] = -iso.score_samples(valid[FEATURES])
scores_test["Isolation forest (no labels)"] = -iso.score_samples(test[FEATURES])
for name, m in [("Logistic regression", logistic()), ("Gradient boosting", boosting())]:
    m.fit(train[FEATURES], train.Class)
    models[name] = m
    scores_valid[name] = m.predict_proba(valid[FEATURES])[:, 1]
    scores_test[name] = m.predict_proba(test[FEATURES])[:, 1]

rows = []
for name in scores_test:
    thr = best_threshold(valid.Class, valid.Amount, scores_valid[name])   # tuned on validation only
    rows.append(summarise(name, test.Class, test.Amount, scores_test[name], thr))
no_model = cost(test.Class, test.Amount, np.zeros(len(test), bool))
rows.append({"model": "No model (refund every fraud)", "alerts": 0, "frauds_caught": 0,
             "frauds_total": int(test.Class.sum()), "fraud_value_stopped": 0.0, "total_cost": no_model})
res = pd.DataFrame(rows)
res["saving_vs_no_model"] = no_model - res.total_cost
res.round(4).to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
print(res.round(3).to_string())

best = "Gradient boosting"
thr_best = best_threshold(valid.Class, valid.Amount, scores_valid[best])

# ---- Figure 1: precision-recall curves
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for (name, s), col in zip(scores_test.items(), [GREY, TERRA, TEAL]):
    c = pr_curve(test.Class, s)
    ap = res.set_index("model").loc[name, "pr_auc"]
    ax.step(c.recall, c.precision, where="post", color=col, lw=2, label=f"{name} (PR-AUC {ap:.2f})")
ax.axhline(test.Class.mean(), color="k", lw=0.8, ls=":", label=f"Random ({test.Class.mean():.2%} fraud)")
ax.set_xlabel("Recall (share of frauds caught)"); ax.set_ylabel("Precision (share of alerts that are fraud)")
ax.set_title("Precision-recall on the latest 20% of transactions"); ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "pr_curves.png", dpi=160); plt.close(fig)

# ---- Figure 2: total cost as the number of alerts grows (test data, both supervised models)
# Cost is computed exactly for every k (review the k highest-scored transactions), so the curve
# has no threshold-grid artefacts.
K = 400
y_t, a_t = test.Class.to_numpy(), test.Amount.to_numpy()
fig, ax = plt.subplots(figsize=(7.4, 4.2))
for name, col in [("Logistic regression", TERRA), ("Gradient boosting", TEAL)]:
    o = np.argsort(-scores_test[name])
    stopped = np.cumsum(np.where(y_t[o] == 1, a_t[o], 0.0))[:K]
    k = np.arange(1, K + 1)
    curve = (no_model - stopped + REVIEW_COST * k) / 1e3
    ax.plot(k, curve, color=col, lw=2, label=name)
    ch = res.set_index("model").loc[name]
    ax.plot(ch.alerts, ch.total_cost / 1e3, "o", color=col, ms=8, mec="white", mew=1.5)
ax.axhline(no_model / 1e3, color=GREY, ls="--", lw=1.2)
ax.text(K, no_model / 1e3 + 0.12, f"No model: \u20ac{no_model/1e3:,.1f}k of fraud refunded", ha="right", va="bottom", fontsize=8.5, color=GREY)
ax.text(K, 0.35, "Dots: threshold chosen on the validation period.\n"
        "Steps down: a fraud is caught (big steps = big frauds). Rising line: each extra alert costs \u20ac5.\n"
        "Logistic regression's first ~20 alerts are mostly \u20ac1 'card-testing' frauds, so its cost barely moves at first.",
        ha="right", va="bottom", fontsize=7.6, color=GREY)
ax.set_xlim(0, K); ax.set_ylim(0, no_model / 1e3 * 1.18)
ax.set_xlabel("Number of transactions flagged for review (riskiest first)")
ax.set_ylabel("Fraud losses + review costs (\u20ac thousands)")
ax.set_title(f"Total cost on the test period (\u20ac{REVIEW_COST:.0f} per review)")
ax.legend(frameon=False, fontsize=8.5, loc="center right")
fig.tight_layout(); fig.savefig(FIG / "cost_curve.png", dpi=160); plt.close(fig)

# ---- Figure 3: frauds caught by review budget
budgets = [0.0005, 0.001, 0.002, 0.005, 0.01]
from fraudlens.evaluate import recall_at_budget
fig, ax = plt.subplots(figsize=(7, 3.8))
w = 0.26
for i, (name, col) in enumerate(zip(scores_test, [GREY, TERRA, TEAL])):
    r = [100 * recall_at_budget(test.Class, scores_test[name], b) for b in budgets]
    ax.bar(np.arange(len(budgets)) + (i - 1) * w, r, w, color=col, label=name)
ax.set_xticks(range(len(budgets)), [f"Top {b:.2%}" for b in budgets])
ax.set_ylabel("% of frauds caught"); ax.set_ylim(0, 100)
ax.set_title("If analysts can only review the riskiest X% of transactions")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "recall_budget.png", dpi=160); plt.close(fig)

# ---- Figure 4: what drives the model (permutation importance on validation, PR-AUC)
pi = permutation_importance(models[best], valid[FEATURES], valid.Class, scoring="average_precision",
                            n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.barh(imp.index, imp.values, color=NAVY)
ax.set_xlabel("Drop in PR-AUC when the feature is shuffled"); ax.set_title("Most important features")
fig.tight_layout(); fig.savefig(FIG / "importance.png", dpi=160); plt.close(fig)

summary = {"rows": int(len(df)), "frauds": int(df.Class.sum()), "fraud_rate": float(df.Class.mean()),
           "split": {k: {"rows": len(v), "frauds": int(v.Class.sum())} for k, v in
                     [("train", train), ("valid", valid), ("test", test)]},
           "review_cost_eur": REVIEW_COST, "results": res.round(4).to_dict(orient="records"),
           "top_features": imp.sort_values(ascending=False).round(4).to_dict()}
(ROOT / "reports" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

# ---- Uncertainty: only 75 frauds in the test period, so bootstrap the saving
rng = np.random.default_rng(42)
y, a = test.Class.to_numpy(), test.Amount.to_numpy()
thr_by = {n: best_threshold(valid.Class, valid.Amount, scores_valid[n]) for n in scores_test}
boot = {n: [] for n in scores_test}
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    base = cost(y[i], a[i], np.zeros(len(i), bool))
    for n in scores_test:
        boot[n].append(base - cost(y[i], a[i], scores_test[n][i] >= thr_by[n]))
ci = {n: {"saving": float(np.median(v)), "low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5)),
          "pct_of_losses_avoided": float(100 * (no_model - res.set_index("model").loc[n, "total_cost"]) / no_model)}
      for n, v in boot.items()}
diff = np.array(boot["Gradient boosting"]) - np.array(boot["Logistic regression"])
ci["boosting_minus_logistic"] = {"median": float(np.median(diff)), "low": float(np.percentile(diff, 2.5)),
                                 "high": float(np.percentile(diff, 97.5))}

# ---- Sensitivity to the review-cost assumption (threshold re-tuned on validation each time)
sens = []
for rc in [1.0, 2.0, 5.0, 10.0, 20.0]:
    for n in ["Logistic regression", "Gradient boosting"]:
        t_ = best_threshold(valid.Class, valid.Amount, scores_valid[n], rc)
        c_ = cost(test.Class, test.Amount, scores_test[n] >= t_, rc)
        sens.append({"review_cost": rc, "model": n, "alerts": int((scores_test[n] >= t_).sum()),
                     "total_cost": c_, "pct_of_losses_avoided": 100 * (no_model - c_) / no_model})
sens = pd.DataFrame(sens)
sens.round(2).to_csv(ROOT / "reports" / "review_cost_sensitivity.csv", index=False)
summary.update({"bootstrap": ci, "review_cost_sensitivity": sens.round(3).to_dict(orient="records"),
                "no_model_cost": no_model})
(ROOT / "reports" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(ci, indent=1)); print(sens.round(1).to_string())
