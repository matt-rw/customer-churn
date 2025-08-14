# Customer Churn Prediction (PyTorch Neural Network)

**A PyTorch-based neural network for binary classification of customer churn.**

## Overview

This project implements a neural network model using PyTorch to predict customer churn. It includes training, evaluation, inference, and saved model artifacts, providing a modular foundation for further experimentation or integration.

---

##  Directory Structure

customer-churn/
├── data/ # Sample data
├── churn/ # Data preprocessing and PyTorch model
├── train.py # Training script for model development
├── evaluate.py # Model evaluation on holdout set
├── infer.py # Inference utility for single predictions
├── churn_model.pt # Trained PyTorch model checkpoint
├── config.py # Configuration settings (paths, hyperparameters)
└── requirements.txt # Python dependencies

---

##  Getting Started

### Prerequisites
- Python 3.9+
- Recommended: virtualenv or `venv`

### Install Dependencies

```bash
git clone https://github.com/matt-rw/customer-churn.git
cd customer-churn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Training the Model
```bash
python train.py
```

### 2. Evaluating the Model
```bash
python evaluate.py
```

Generates key performance metrics: accuracy, confusion matrix, and classification report.

### 3. Running Inference
Predicts churn probability for a single customer sample (not implemented).

## Additional Enhancements (Future Directions)

* Adjust hyperparameters (learning rate, layers, optimizer) via config.py or direct flags.
* Implement advanced evaluation dashboards (e.g., using plotly or matplotlib)
* Add model versioning & metrics tracking (e.g., MLflow, Weights & Biases)
* Extend to REST API using FastAPI or Flask for production integration
* Scale inference through Docker or a lightweight cloud deployment (Render or Fly.io)
