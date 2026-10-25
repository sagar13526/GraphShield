"""Interactive Streamlit dashboard for the synthetic UPI-style fraud project."""
from pathlib import Path
import json

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

st.set_page_config(page_title="GraphShield | Fraud Analytics", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.07); padding: 14px 16px; border-radius: 12px;}
.small-note {font-size: .88rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ GraphShield")
st.caption("Graph-based transaction fraud exploration · Educational prototype")
st.warning("Synthetic data only. This dashboard is not connected to a bank or payment network, and its scores are not validated for real-world use.")

required = [DATA / "accounts.csv", DATA / "transactions.csv", RESULTS / "metrics.json", RESULTS / "test_predictions.npz"]
missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
if missing:
    st.error("Project outputs are missing: " + ", ".join(missing))
    st.info("From the project folder, run `run.bat` first. After training completes, start the dashboard with `run_dashboard.bat`.")
    st.stop()

accounts = pd.read_csv(DATA / "accounts.csv")
transactions = pd.read_csv(DATA / "transactions.csv")
metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
pred_file = np.load(RESULTS / "test_predictions.npz")

# Only accounts in the held-out test split have saved predictions.
test_idx = pred_file["test_idx"].astype(int)
account_ids = accounts["account_id"].astype(str).to_numpy()
predictions = pd.DataFrame({
    "account_id": account_ids[test_idx],
    "graphsage_score": pred_file["graphsage"].astype(float),
    "logistic_score": pred_file["logistic_regression"].astype(float),
    "test_index": test_idx,
})
predictions = predictions.merge(accounts[["account_id", "fraud_label", "ring_id"]], on="account_id", how="left")
predictions["predicted_fraud"] = (predictions["graphsage_score"] >= 0.5).astype(int)
predictions["risk_band"] = pd.cut(predictions["graphsage_score"], bins=[-0.001, 0.3, 0.7, 1.001], labels=["Lower", "Elevated", "High"])

with st.sidebar:
    st.header("Explore")
    page = st.radio("Dashboard section", ["Overview", "Account explorer", "Transaction network", "Data browser"])
    st.divider()
    st.caption("Evaluation split")
    st.write(f"**{len(predictions)}** held-out test accounts")
    st.write("Risk score threshold: **0.50**")
    st.caption("Only test-set accounts have saved prediction scores. Other accounts are not assigned fabricated scores.")

if page == "Overview":
    st.subheader("Model performance")
    ds = metrics.get("dataset", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accounts", f"{len(accounts):,}")
    m2.metric("Transfers", f"{len(transactions):,}")
    m3.metric("Simulated fraud accounts", f"{int(accounts['fraud_label'].sum()):,}")
    m4.metric("Test accounts", f"{len(predictions):,}")

    model_rows = []
    for key, label in [("graphsage", "GraphSAGE"), ("logistic_regression", "Logistic Regression")]:
        m = metrics[key]
        model_rows.append({"Model": label, "ROC-AUC": m["roc_auc"], "Average Precision": m["average_precision"], "Precision": m["precision_at_0_5"], "Recall": m["recall_at_0_5"], "F1-score": m["f1_at_0_5"]})
    model_df = pd.DataFrame(model_rows)
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Held-out test metrics")
        st.dataframe(model_df.style.format({c: "{:.3f}" for c in model_df.columns if c != "Model"}), use_container_width=True, hide_index=True)
        st.caption("Precision, recall and F1 use a fixed probability threshold of 0.50.")
    with right:
        metric_choice = st.selectbox("Compare metric", ["ROC-AUC", "Average Precision", "Precision", "Recall", "F1-score"])
        chart = px.bar(model_df, x="Model", y=metric_choice, color="Model", range_y=[0, 1], text_auto=".3f", title=f"{metric_choice} comparison")
        chart.update_layout(showlegend=False, yaxis_title="Score", xaxis_title="")
        st.plotly_chart(chart, use_container_width=True)

    st.markdown("#### GraphSAGE confusion matrix")
    cm = np.array(metrics["graphsage"]["confusion_matrix_tn_fp_fn_tp"], dtype=int)
    cm_df = pd.DataFrame(cm, index=["Actual normal", "Actual simulated fraud"], columns=["Predicted normal", "Predicted fraud"])
    st.dataframe(cm_df, use_container_width=True)
    st.caption("Confusion matrix counts are from the held-out test set, not from the full dataset.")

    st.markdown("#### What this experiment says")
    st.write("GraphSAGE combines account features with graph-neighborhood information. Logistic Regression is the baseline that uses account features without neighborhood aggregation. Results apply only to this generated dataset and this split.")

elif page == "Account explorer":
    st.subheader("Account risk explorer")
    st.write("Choose an account from the held-out test set. Scores are model outputs; the simulated label is shown separately for educational evaluation.")
    choices = predictions.sort_values("graphsage_score", ascending=False)["account_id"].tolist()
    selected = st.selectbox("Test account", choices)
    row = predictions.loc[predictions.account_id == selected].iloc[0]
    a = accounts.loc[accounts.account_id == selected].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GraphSAGE risk score", f"{row.graphsage_score:.1%}")
    c2.metric("Risk band", str(row.risk_band))
    c3.metric("Predicted at 0.50", "Flagged" if row.predicted_fraud else "Not flagged")
    c4.metric("Simulated ground truth", "Fraud-labeled" if int(row.fraud_label) else "Normal-labeled")
    st.progress(float(np.clip(row.graphsage_score, 0, 1)), text="GraphSAGE estimated score (0–100%)")

    st.markdown("#### Account activity features")
    feature_cols = [c for c in accounts.columns if c not in ["account_id", "fraud_label", "ring_id"]]
    st.dataframe(pd.DataFrame({"Feature": feature_cols, "Value": [a[c] for c in feature_cols]}), use_container_width=True, hide_index=True)

    # Show direct counterparties based on the transaction table.
    linked = transactions[(transactions.source == selected) | (transactions.destination == selected)].copy()
    st.markdown("#### Direct transfers")
    if linked.empty:
        st.info("No transactions were found for this account.")
    else:
        st.metric("Direct transfer records", f"{len(linked):,}")
        st.dataframe(linked.sort_values("amount", ascending=False).head(100), use_container_width=True, hide_index=True)
    st.caption("Risk scores are available only for held-out test accounts in the saved prediction file. A high score is not proof of fraud.")

elif page == "Transaction network":
    st.subheader("Transaction network")
    st.write("Each node is an account and each directed edge represents a simulated transfer. Node colors indicate the synthetic label, not a real-world finding.")
    max_nodes = st.slider("Accounts to display", min_value=20, max_value=100, value=55, step=5)
    show_edges = st.slider("Maximum edges to display", min_value=50, max_value=500, value=220, step=25)

    # Select a readable subset: highest-degree accounts plus high-score test accounts.
    degree = pd.concat([transactions.source, transactions.destination]).value_counts()
    high_degree = degree.head(max_nodes // 2).index.tolist()
    high_score = predictions.sort_values("graphsage_score", ascending=False).head(max_nodes - len(high_degree))["account_id"].tolist()
    selected_nodes = list(dict.fromkeys(high_degree + high_score))[:max_nodes]
    # Fill any remaining slots from accounts with the largest transaction degree.
    if len(selected_nodes) < max_nodes:
        for aid in degree.index:
            if aid not in selected_nodes:
                selected_nodes.append(aid)
            if len(selected_nodes) >= max_nodes:
                break
    sub_edges = transactions[transactions.source.isin(selected_nodes) & transactions.destination.isin(selected_nodes)].copy()
    if len(sub_edges) > show_edges:
        sub_edges = sub_edges.sample(show_edges, random_state=42)

    G = nx.DiGraph()
    G.add_nodes_from(selected_nodes)
    G.add_edges_from(zip(sub_edges.source, sub_edges.destination))
    positions = nx.spring_layout(G, seed=42, k=0.6, iterations=35)
    edge_x, edge_y = [], []
    for src, dst in G.edges():
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.6), opacity=0.35, hoverinfo="skip", name="Transfers"))
    pred_map = predictions.set_index("account_id")["graphsage_score"].to_dict()
    label_map = accounts.set_index("account_id")["fraud_label"].to_dict()
    ring_map = accounts.set_index("account_id")["ring_id"].to_dict()
    degree_map = degree.to_dict()
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in selected_nodes:
        x, y = positions[node]
        node_x.append(x); node_y.append(y)
        score = pred_map.get(node, None)
        score_text = f"{score:.1%}" if score is not None else "Not in held-out test set"
        node_text.append(f"{node}<br>Simulated label: {'Fraud' if label_map.get(node, 0) else 'Normal'}<br>GraphSAGE score: {score_text}<br>Transfers: {degree_map.get(node, 0)}<br>Ring ID: {ring_map.get(node, -1)}")
        node_color.append("Simulated fraud" if label_map.get(node, 0) else "Normal")
        node_size.append(10 + min(degree_map.get(node, 0), 25) * 0.5)
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", text=node_text, hoverinfo="text", marker=dict(size=node_size, color=[1 if c == "Simulated fraud" else 0 for c in node_color], colorscale=[[0, "#60a5fa"], [1, "#f97316"]], line=dict(width=0.5, color="white")), name="Accounts"))
    fig.update_layout(height=650, showlegend=False, margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The network view samples a subset of accounts and edges for readability. Hover over a node to inspect details. Accounts outside the test split do not have saved model scores.")

else:
    st.subheader("Data browser")
    tab1, tab2, tab3 = st.tabs(["Accounts", "Transactions", "Test predictions"])
    with tab1:
        st.caption("Account-level features and synthetic labels. Labels are part of the generated benchmark, not live risk assessments.")
        st.dataframe(accounts, use_container_width=True, hide_index=True)
        st.download_button("Download accounts.csv", accounts.to_csv(index=False).encode("utf-8"), "accounts.csv", "text/csv")
    with tab2:
        st.dataframe(transactions, use_container_width=True, hide_index=True)
        st.download_button("Download transactions.csv", transactions.to_csv(index=False).encode("utf-8"), "transactions.csv", "text/csv")
    with tab3:
        st.dataframe(predictions.sort_values("graphsage_score", ascending=False), use_container_width=True, hide_index=True)
        st.download_button("Download test predictions.csv", predictions.to_csv(index=False).encode("utf-8"), "test_predictions.csv", "text/csv")

st.divider()
st.caption("GraphShield · Synthetic educational benchmark · Do not use for real payment decisions")
