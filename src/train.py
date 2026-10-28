"""Train and evaluate the ChurnNet deep learning model on IBM Telco churn data.

Usage:
    python src/train.py --data data/telco_churn.csv --out results
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve, accuracy_score)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from data import TARGET, Preprocessor, add_features, load_raw, split  # noqa: E402
from model import ChurnNet  # noqa: E402


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, x_cont, x_cat):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x_cont), torch.from_numpy(x_cat))).numpy()


def train_one(seed, pp, tr, va, y_tr, y_va, args):
    set_seed(seed)
    model = ChurnNet(pp.cardinalities, tr[0].shape[1], hidden=args.hidden,
                     n_blocks=args.blocks, dropout=args.dropout)
    pos_weight = torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum(), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    dl = DataLoader(TensorDataset(torch.from_numpy(tr[0]), torch.from_numpy(tr[1]),
                                  torch.from_numpy(y_tr.astype(np.float32))),
                    batch_size=args.batch, shuffle=True, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(dl))
    best_auc, best_state, bad, history = -1, None, 0, []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xc, xk, y in dl:
            opt.zero_grad()
            loss = loss_fn(model(xc, xk), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        p_va = predict(model, *va)
        val_auc = roc_auc_score(y_va, p_va)
        history.append({"epoch": epoch + 1, "train_loss": total / (len(dl) * args.batch),
                        "val_auc": val_auc})
        if val_auc > best_auc + 1e-4:
            best_auc, bad = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best_auc, history


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "threshold": thr}


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 181)
    return float(grid[np.argmax([f1_score(y, (p >= t).astype(int)) for t in grid])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telco_churn.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5, help="ensemble size")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = add_features(load_raw(args.data))
    train, val, test = split(df)
    pp = Preprocessor().fit(train)
    tr, va, te = pp.transform(train), pp.transform(val), pp.transform(test)
    y_tr, y_va, y_te = train[TARGET].values, val[TARGET].values, test[TARGET].values
    print(f"train={len(train)} val={len(val)} test={len(test)} churn_rate={df[TARGET].mean():.3f}")

    # --- Deep ensemble ---------------------------------------------------
    models, histories = [], []
    for s in range(args.seeds):
        m, auc, h = train_one(s, pp, tr, va, y_tr, y_va, args)
        models.append(m)
        histories.append(h)
        print(f"seed {s}: best val AUC={auc:.4f} after {len(h)} epochs")
    p_va = np.mean([predict(m, *va) for m in models], axis=0)
    p_te = np.mean([predict(m, *te) for m in models], axis=0)
    thr = best_threshold(y_va, p_va)  # tuned on validation only
    dl_metrics = metrics(y_te, p_te, thr)

    # --- Baseline: logistic regression on one-hot features ---------------
    X = pd.get_dummies(df.drop(columns=["customerID", TARGET]), drop_first=True)

    def onehot(frame):
        return pd.get_dummies(frame.drop(columns=["customerID", TARGET]), drop_first=True) \
            .reindex(columns=X.columns, fill_value=0).astype(float).values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(onehot(train))
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(sc.transform(onehot(train)), y_tr)
    p_va_lr = lr.predict_proba(sc.transform(onehot(val)))[:, 1]
    p_te_lr = lr.predict_proba(sc.transform(onehot(test)))[:, 1]
    lr_metrics = metrics(y_te, p_te_lr, best_threshold(y_va, p_va_lr))

    results = {"deep_ensemble": dl_metrics, "logistic_regression_baseline": lr_metrics,
               "n_test": int(len(test)), "config": vars(args)}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ["deep_ensemble", "logistic_regression_baseline"]}, indent=2))

    # --- Save model ------------------------------------------------------
    torch.save({"state_dicts": [m.state_dict() for m in models],
                "cardinalities": pp.cardinalities, "threshold": thr,
                "config": vars(args)}, os.path.join(args.out, "churnnet_ensemble.pt"))

    # --- Plots -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    for name, p in [("Deep ensemble", p_te), ("Logistic regression", p_te_lr)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend()
    ax = axes[0, 1]
    for name, p in [("Deep ensemble", p_te), ("Logistic regression", p_te_lr)]:
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_te, p):.3f})")
    ax.axhline(y_te.mean(), color="k", ls="--", lw=0.8, label="No-skill")
    ax.set(title="Precision–recall curve (test set)", xlabel="Recall", ylabel="Precision")
    ax.legend()
    ax = axes[1, 0]
    for i, h in enumerate(histories):
        ax.plot([r["epoch"] for r in h], [r["val_auc"] for r in h], alpha=0.8, label=f"seed {i}")
    ax.set(title="Validation ROC-AUC per epoch", xlabel="Epoch", ylabel="Val AUC")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    cm = confusion_matrix(y_te, (p_te >= thr).astype(int))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set(title=f"Confusion matrix (threshold={thr:.2f})", xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"],
           xlabel="Predicted", ylabel="Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "evaluation.png"), dpi=130)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
