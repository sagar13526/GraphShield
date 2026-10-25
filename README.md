# GraphShield — Graph-Based UPI Fraud Detection

GraphShield is a student machine learning project that identifies potentially suspicious UPI-style transactions by analyzing relationships between accounts and transactions using graph-based learning.

The project uses synthetic transaction data to demonstrate how graph neural networks can help detect suspicious transaction patterns.

## Features

- Generates synthetic account and transaction data.
- Represents transaction relationships as a graph.
- Uses a GraphSAGE-style graph learning model to classify suspicious transactions.
- Compares the graph-based model with a Logistic Regression baseline.
- Displays predictions, evaluation metrics, and transaction network visualizations.
- Provides an interactive dashboard built with Streamlit.

## Technologies Used

- Python
- PyTorch
- Scikit-learn
- Pandas and NumPy
- Matplotlib and NetworkX
- Streamlit

## Project Structure

```text
GraphShield/
├── app.py
├── requirements.txt
├── run.bat
├── run_dashboard.bat
├── README.md
├── data/
│   ├── accounts.csv
│   ├── transactions.csv
│   └── graph.npz
├── results/
│   ├── graphsage_model.pt
│   ├── metrics.json
│   ├── model_comparison.png
│   ├── test_predictions.npz
│   └── transaction_network.png
└── src/
    ├── generate_data.py
    ├── main.py
    ├── model.py
    ├── train.py
    └── visualize.py
```

## How It Works

1. **Data Generation:** Synthetic accounts and transactions are generated for the experiment.
2. **Graph Construction:** Accounts and transaction relationships are represented as a graph.
3. **Model Training:** A GraphSAGE-style model learns patterns from the graph to classify transactions.
4. **Baseline Comparison:** Logistic Regression is used as a baseline model.
5. **Evaluation:** The models are evaluated using classification metrics.
6. **Visualization:** The dashboard presents results and transaction network visualizations.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sagar13526/GraphShield.git
cd GraphShield
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

To launch the dashboard, run:

```bash
streamlit run app.py
```

Alternatively, if the batch files are configured for your environment, you can use `run_dashboard.bat` or `run.bat`.

If required data and model files are missing, run the relevant data-generation and training scripts in the `src` folder before launching the dashboard.

## Model Evaluation

GraphShield compares the graph-based model with Logistic Regression using classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score

The project also includes visual comparisons and transaction network plots. Actual performance should be checked using the generated evaluation results.

## Limitations

- The project uses synthetic data, not real bank or UPI transaction records.
- Its results do not establish real-world fraud detection performance.
- The model is an educational prototype and is not intended for production banking systems.
- Performance depends on the generated data, features, and training setup.

## Future Improvements

- Evaluate the model on suitable real-world datasets where permitted.
- Improve graph features and model architecture.
- Test the system on larger transaction networks.
- Add more detailed transaction explanations and monitoring.
- Explore methods for handling changing fraud patterns.

## Purpose

This project demonstrates the use of graph-based machine learning for transaction fraud detection and provides practical experience with data generation, model training, model comparison, and dashboard development.

**Note:** GraphShield is an educational project. It does not connect to a real UPI system or make decisions about actual financial transactions.
