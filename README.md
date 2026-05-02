# Telco Customer Churn - Kedro + MLflow Demo

A simple end-to-end machine learning project for predicting customer churn.  
This project demonstrates how to structure a production-style ML workflow with:

- Structured pipelines using Kedro  
- Experiment tracking using MLflow  
- Data validation with Great Expectations  
- Hyperparameter tuning with Optuna  
- Threshold tuning to optimize recall  
- Model calibration using Venn-Abers calibration  
- Model explainability with SHAP  


## Setup & Run

### Prerequisites

- Python >= 3.10  
- uv (Python package manager)

---

### 1. Clone the repository

```bash
git clone https://github.com/shafayetShafee/telco-customer-churn-mlops.git
cd telco-customer-churn-mlops
```

### 2. Install dependencies

```bash
uv sync --all-extras
```

### 3. Train the model

```bash
kedro run --pipeline=train
```

### 4. Run inference

```bash
kedro run --pipeline=user
```


## Kedro Viz

Run the following from the project root directory, 

```bash
kedro viz run -a --include-hooks
```

## ETL Pipeline

![ETL Pipeline](imgs/kedro-etl-pipeline.png)


## Train Pipeline

![Train Pipeline](imgs/kedro-train-pipeline.png)


## User Pipeline

![User Pipeline](imgs/kedro-user-pipeline.png)


