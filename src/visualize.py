"""Create a graph illustration and model-comparison chart."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def main():
    graph = np.load(DATA / "graph.npz")
    edges = graph["edges"]
    labels = graph["labels"]
    ring_ids = graph["ring_ids"]
    rng = np.random.default_rng(42)

    # Lightweight force-like illustration using circular positions, no extra graph library.
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = rng.uniform(0.7, 1.3, size=n)
    xy = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    fig, ax = plt.subplots(figsize=(9, 7))
    # Plot a manageable sample of edges so the visualization stays readable.
    chosen = edges[rng.choice(len(edges), size=min(450, len(edges)), replace=False)]
    for src, dst in chosen:
        ax.plot([xy[src, 0], xy[dst, 0]], [xy[src, 1], xy[dst, 1]],
                linewidth=0.35, alpha=0.12)
    normal = labels == 0
    fraud = labels == 1
    ax.scatter(xy[normal, 0], xy[normal, 1], s=18, label="Normal account")
    ax.scatter(xy[fraud, 0], xy[fraud, 1], s=38, marker="x", label="Simulated fraud account")
    ax.set_title("Synthetic UPI-Style Transaction Network")
    ax.set_axis_off()
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "transaction_network.png", dpi=180)
    plt.close(fig)

    metrics_path = RESULTS / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        names = ["GraphSAGE", "Logistic Regression"]
        aucs = [m["graphsage"]["roc_auc"], m["logistic_regression"]["roc_auc"]]
        aps = [m["graphsage"]["average_precision"], m["logistic_regression"]["average_precision"]]
        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, aucs, width, label="ROC-AUC")
        ax.bar(x + width/2, aps, width, label="Average Precision")
        ax.set_xticks(x, names)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison on Synthetic Test Set")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "model_comparison.png", dpi=180)
        plt.close(fig)
    print(f"Saved visualizations in {RESULTS}")

if __name__ == "__main__":
    main()
