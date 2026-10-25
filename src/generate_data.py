"""Generate a reproducible synthetic account-transfer graph for learning."""
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

def generate(seed=42, n_accounts=300, n_rings=8, ring_size=6):
    rng = np.random.default_rng(seed)
    labels = np.zeros(n_accounts, dtype=int)
    ring_ids = np.full(n_accounts, -1, dtype=int)

    # Reserve a small set of accounts for simulated coordinated fraud rings.
    ring_accounts = []
    available = np.arange(n_accounts)
    rng.shuffle(available)
    cursor = 0
    for rid in range(n_rings):
        members = available[cursor:cursor + ring_size].tolist()
        cursor += ring_size
        for node in members:
            labels[node] = 1
            ring_ids[node] = rid
        ring_accounts.append(members)

    edges = []
    amounts = []
    # Normal activity: mostly random transfers, with a few high-volume accounts.
    activity = rng.lognormal(mean=1.2, sigma=0.65, size=n_accounts)
    for _ in range(2200):
        src = int(rng.integers(0, n_accounts))
        # activity-weighted destination selection creates realistic degree variation
        probs = activity / activity.sum()
        dst = int(rng.choice(n_accounts, p=probs))
        while dst == src:
            dst = int(rng.choice(n_accounts, p=probs))
        amount = float(np.clip(rng.lognormal(4.0, 1.0), 10, 50000))
        edges.append((src, dst))
        amounts.append(amount)

    # Fraud rings transfer money among members in repeated coordinated patterns.
    for members in ring_accounts:
        for _ in range(18):
            src = int(rng.choice(members))
            dst = int(rng.choice(members))
            while dst == src:
                dst = int(rng.choice(members))
            amount = float(np.clip(rng.lognormal(5.0, 0.7), 100, 50000))
            edges.append((src, dst))
            amounts.append(amount)

    edges = np.asarray(edges, dtype=int)
    amounts = np.asarray(amounts, dtype=float)
    n = n_accounts
    out_degree = np.bincount(edges[:, 0], minlength=n).astype(float)
    in_degree = np.bincount(edges[:, 1], minlength=n).astype(float)
    out_sum = np.bincount(edges[:, 0], weights=amounts, minlength=n)
    in_sum = np.bincount(edges[:, 1], weights=amounts, minlength=n)
    out_avg = out_sum / np.maximum(out_degree, 1)
    in_avg = in_sum / np.maximum(in_degree, 1)

    # Count distinct counterparties for each account.
    neighbors = [set() for _ in range(n)]
    for src, dst in edges:
        neighbors[src].add(int(dst))
        neighbors[dst].add(int(src))
    unique_neighbors = np.array([len(s) for s in neighbors], dtype=float)

    # Account-level features; labels are deliberately NOT included as features.
    features = np.column_stack([
        np.log1p(out_degree),
        np.log1p(in_degree),
        np.log1p(out_avg),
        np.log1p(in_avg),
        np.log1p(unique_neighbors),
        np.log1p(out_sum + in_sum),
        out_degree / np.maximum(in_degree, 1),
    ]).astype(float)

    with (DATA / "accounts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["account_id", "fraud_label", "ring_id",
                         "log_out_degree", "log_in_degree", "log_avg_out_amount",
                         "log_avg_in_amount", "log_unique_counterparties",
                         "log_total_amount", "out_in_degree_ratio"])
        for i in range(n):
            writer.writerow([f"ACC{i:04d}", int(labels[i]), int(ring_ids[i]), *features[i].tolist()])

    with (DATA / "transactions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "destination", "amount"])
        for (src, dst), amount in zip(edges, amounts):
            writer.writerow([f"ACC{src:04d}", f"ACC{dst:04d}", round(float(amount), 2)])

    np.savez(DATA / "graph.npz", features=features, labels=labels, edges=edges,
             amounts=amounts, ring_ids=ring_ids)
    print(f"Generated {n} accounts and {len(edges)} transfers.")
    print(f"Simulated fraud accounts: {int(labels.sum())} ({labels.mean():.1%}).")
    print(f"Saved CSV and graph files in: {DATA}")

if __name__ == "__main__":
    generate()
