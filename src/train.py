"""Train GraphSAGE and a Logistic Regression baseline, then save metrics."""
from pathlib import Path
import json
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from model import GraphSAGE

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42

def metrics_for(name, y_true, probs):
    pred = (probs >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "average_precision": float(average_precision_score(y_true, probs)),
        "precision_at_0_5": float(precision_score(y_true, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    graph = np.load(DATA / "graph.npz")
    X = graph["features"].astype(np.float32)
    y = graph["labels"].astype(int)
    edges = graph["edges"].astype(int)
    n = len(y)

    all_idx = np.arange(n)
    train_idx, temp_idx = train_test_split(
        all_idx, test_size=0.4, random_state=SEED, stratify=y)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=SEED, stratify=y[temp_idx])

    scaler = StandardScaler()
    X_scaled = np.zeros_like(X)
    X_scaled[train_idx] = scaler.fit_transform(X[train_idx])
    X_scaled[val_idx] = scaler.transform(X[val_idx])
    X_scaled[test_idx] = scaler.transform(X[test_idx])

    # Undirected, self-looped, row-normalized adjacency for neighborhood aggregation.
    A = np.zeros((n, n), dtype=np.float32)
    for src, dst in edges:
        A[src, dst] = 1.0
        A[dst, src] = 1.0
    A += np.eye(n, dtype=np.float32)
    A /= np.maximum(A.sum(axis=1, keepdims=True), 1.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    x_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    a_t = torch.tensor(A, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    model = GraphSAGE(input_dim=X.shape[1], hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    # Handle class imbalance using positive-class weighting based on training data.
    positives = max(int(y[train_idx].sum()), 1)
    negatives = max(len(train_idx) - positives, 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device))

    best_val_loss = float("inf")
    best_state = None
    patience, wait = 18, 0
    for epoch in range(1, 151):
        model.train()
        optimizer.zero_grad()
        logits = model(x_t, a_t)
        loss = loss_fn(logits[train_t], y_t[train_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_t, a_t)
            val_loss = float(loss_fn(val_logits[val_t], y_t[val_t]).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train loss {loss.item():.4f} | val loss {val_loss:.4f}")
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        gnn_probs = torch.sigmoid(model(x_t, a_t)).detach().cpu().numpy()[test_idx]

    # Baseline uses only the account's own features, without graph neighborhood aggregation.
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(X_scaled[train_idx], y[train_idx])
    lr_probs = lr.predict_proba(X_scaled[test_idx])[:, 1]

    results = {
        "project": "UPI Fraud Detection with GraphSAGE",
        "dataset": {
            "type": "synthetic",
            "accounts": int(n),
            "transfers": int(len(edges)),
            "fraud_accounts": int(y.sum()),
            "train_accounts": int(len(train_idx)),
            "validation_accounts": int(len(val_idx)),
            "test_accounts": int(len(test_idx)),
            "seed": SEED,
        },
        "graphsage": metrics_for("GraphSAGE", y[test_idx], gnn_probs),
        "logistic_regression": metrics_for("Logistic Regression", y[test_idx], lr_probs),
        "note": "Synthetic educational benchmark only; not validated for real payment systems."
    }
    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), RESULTS / "graphsage_model.pt")
    np.savez(RESULTS / "test_predictions.npz", y_true=y[test_idx],
             graphsage=gnn_probs, logistic_regression=lr_probs, test_idx=test_idx)
    print("\n=== Test-set comparison ===")
    for key in ("graphsage", "logistic_regression"):
        m = results[key]
        print(f"{m['model']:20s} ROC-AUC={m['roc_auc']:.3f} | AP={m['average_precision']:.3f} "
              f"| Precision={m['precision_at_0_5']:.3f} | Recall={m['recall_at_0_5']:.3f} "
              f"| F1={m['f1_at_0_5']:.3f}")
    print(f"\nSaved metrics to {RESULTS / 'metrics.json'}")
    print("Remember: report only the measured results from this run.")

if __name__ == "__main__":
    main()
