# GraphShield — UPI-Style Transaction Fraud Detection

GraphShield is an educational machine-learning project that explores how graph structure can help identify **simulated suspicious accounts** in a synthetic UPI-style transaction network. It includes a reproducible data generator, a simplified GraphSAGE-style PyTorch model, a Logistic Regression baseline, evaluation outputs, and an interactive Streamlit dashboard.

> **Scope and safety:** This project uses synthetic data only. It is not connected to UPI, Visa, a bank, or any payment network. Scores are not validated for real-world use and must never be used to make actual payment decisions.

## Features

- Generate a synthetic account graph and transfer records with simulated coordinated rings.
- Engineer account-level activity features from incoming/outgoing transactions and counterparties.
- Train a two-layer **GraphSAGE-style** neural network in PyTorch.
- Compare against a Logistic Regression baseline.
- Evaluate on a held-out test split using ROC-AUC, Average Precision, precision, recall, F1, and a confusion matrix.
- Export metrics, test predictions, model weights, CSV data, and static plots.
- Explore metrics, accounts, transaction links, and the graph in a Streamlit dashboard.

## Project structure

```text
GraphShield-UPI-Fraud-Detection/
├── app.py                       # Interactive Streamlit dashboard
├── run.bat                      # Generate data, train, and create plots
├── run_dashboard.bat            # Launch the dashboard
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── LICENSE                      # MIT license
├── .gitignore
├── src/
│   ├── generate_data.py          # Synthetic graph/data generator
│   ├── model.py                  # GraphSAGE-style PyTorch model
│   ├── train.py                  # Training, evaluation, and saved predictions
│   ├── visualize.py              # Static plots
│   └── main.py                   # Runs the full training pipeline
├── data/
│   ├── accounts.csv              # Account features and simulated labels
│   ├── transactions.csv          # Synthetic directed transfers
│   └── graph.npz                 # NumPy arrays used by the model
└── results/
    ├── metrics.json              # Measured metrics from the included run
    ├── test_predictions.npz      # Test labels, scores, and account indices
    ├── graphsage_model.pt        # Saved PyTorch model weights
    ├── model_comparison.png      # Static model comparison plot
    └── transaction_network.png   # Static network illustration
```

The `*.npz` files are binary NumPy archives, not documents. Inspect them using Python/NumPy; open the CSV files in Excel or VS Code. The `.pt` file contains model weights and is not meant to be opened as text. All source code is in the human-readable `.py` files under `src/` and in `app.py`. You can ignore `__pycache__` if Python creates it when you run the project; it is generated automatically and is not included in this ZIP.

## Requirements

- Windows 10/11
- Python 3.11 or newer (your installed Python must be supported by the PyTorch build pip installs)
- Internet access for the first dependency installation
- Approximately 1–2 GB free disk space is recommended for Python packages

The pipeline uses PyTorch, NumPy, scikit-learn, and Matplotlib. The dashboard additionally uses Streamlit, Plotly, pandas, and NetworkX. `requirements.txt` installs all of them.

## Run on Windows

1. Extract the ZIP completely.
2. Open the extracted `GraphShield-UPI-Fraud-Detection` folder.
3. Double-click **`run.bat`**. It installs/checks dependencies, regenerates synthetic data, trains both models, and creates the static plots.
4. When it finishes, double-click **`run_dashboard.bat`**. The dashboard should open in your browser. If it does not, copy the local URL shown in the terminal (usually `http://localhost:8501`) into your browser.

You can also open a terminal in the project folder and run:

```bat
python -m pip install -r requirements.txt
python src\main.py
python -m streamlit run app.py
```

If the commands fail, keep the terminal open and copy the final error lines. Do not run the commands from the parent folder; run them from the folder that contains `run.bat` and `app.py`.

## How the experiment works

1. A seeded generator creates synthetic accounts, transfers, and coordinated simulated fraud rings.
2. Features are computed from account transaction activity. The fraud label and ring ID are not included as input features.
3. Account indices are divided into stratified training, validation, and test splits.
4. A scaler is fit only on the training features and applied to the other splits.
5. The GraphSAGE-style model combines each node's own features with mean-neighbor representations over the graph.
6. Logistic Regression is trained as a feature-only baseline.
7. Metrics are calculated on the held-out test accounts. The saved dashboard risk scores refer only to these test accounts.

This is a **transductive educational setup**: the graph structure and unlabeled node features are available during message passing, while test labels are not used to train the model. This setup is not equivalent to a prospective, time-based evaluation of future transactions. Feature construction from the full generated transfer history can also make the task easier than a real deployment.

## Understanding the outputs

- **ROC-AUC:** ranking quality across classification thresholds.
- **Average Precision:** precision-recall performance, useful when the positive class is less common.
- **Precision / Recall / F1:** reported at a fixed probability threshold of 0.50.
- **Confusion matrix:** true negatives, false positives, false negatives, and true positives on the test split.
- **`data/graph.npz`:** arrays named `features`, `labels`, `edges`, `amounts`, and `ring_ids`.
- **`results/test_predictions.npz`:** arrays named `y_true`, `graphsage`, `logistic_regression`, and `test_idx`.

To inspect the graph archive from the project folder:

```bat
python -c "import numpy as np; d=np.load('data/graph.npz'); print(d.files); [print(k, d[k].shape, d[k].dtype) for k in d.files]"
```

## Limitations and responsible use

- Synthetic fraud rings are simplified and may be easier to detect than real fraud.
- The model is a simplified GraphSAGE-style implementation, not a production-ready fraud system.
- The graph is evaluated transductively; there is no time-based holdout or real-world validation.
- A fixed 0.50 threshold may not suit operational fraud review.
- A real deployment would require approved data access, privacy and security controls, time-based testing, leakage checks, calibration, drift monitoring, explainability, and human review.

## Resume wording

Use measured results from your own `results/metrics.json`, and describe the dataset honestly. For example:

> Built an educational GraphSAGE-style fraud-detection prototype on synthetic transaction graphs; engineered account-level activity features, compared against Logistic Regression, and evaluated with ROC-AUC, Average Precision, precision, recall, and F1.

Do not claim that synthetic data came from Visa, UPI, or a bank, and do not present synthetic benchmark scores as real-world performance.
