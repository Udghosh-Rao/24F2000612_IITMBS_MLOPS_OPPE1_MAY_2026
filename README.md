# Stock Movement Predictor - MLOps OPPE

Roll Number: 24F2000612

## Objective

Predict whether a stock closes higher five minutes after the current observation.

Target:
- 1: close(t+5) > close(t)
- 0: otherwise

## Features

- rolling_avg_10: mean close over the last 10 available observations
- volume_sum_10: total volume over the last 10 available observations
- stock_name: stock entity

Data is sorted chronologically before feature generation.

## Data Versions

Iteration 1 uses v0:
- AARTIIND
- ABCAPITAL

Iteration 2 uses merged v0 + v1:
- AARTIIND
- ABCAPITAL
- ABFRL
- ADANIENT
- ADANIGAS

## DVC

DVC tracks the two raw versions independently:
- data/v0.dvc
- data/v1.dvc

Google Cloud Storage is configured as the DVC remote.

The remote was validated by deleting local data and restoring it with dvc pull.

CI evaluation data is separately versioned:
- ci_data/test.parquet.dvc

MLflow registry state and artifacts are also reproducibly stored with DVC:
- mlflow.db.dvc
- mlruns.dvc

## Feast

Entity:
- stock_name

Feature view:
- rolling_avg_10
- volume_sum_10

Definitions:
- feature_repo/features.py
- feature_repo/feature_store.yaml

The project uses feast apply and Feast historical feature retrieval for point-in-time feature access.

## Training and Evaluation

A chronological 80/20 train/test split is used instead of random splitting.

Iteration 1:
- Train rows: 523,180
- Test rows: 146,164
- Best F1: 0.1133

Iteration 2:
- Train rows: 1,231,159
- Test rows: 365,434
- Best F1: 0.0595

## MLflow

Experiment:
- stock-movement-predictor

Three hyperparameter configurations are evaluated per iteration.

Logged information includes:
- iteration
- hyperparameters
- chronological split
- accuracy
- F1
- model artifacts

Registered model:
- StockMovementPredictor
- Version 1
- Status READY

## Feature Sanity Tests

tests/test_features.py independently validates:
- rolling_avg_10 against raw close values
- volume_sum_10 against raw volume values

## CI and CML

GitHub Actions runs on pushes and pull requests to main.

Pipeline:
1. Install dependencies
2. Authenticate to GCP
3. Pull DVC-pinned raw/test/MLflow data
4. Run feature sanity tests
5. Fetch StockMovementPredictor from MLflow Model Registry
6. Evaluate predictions
7. Generate metrics and confusion matrix
8. Generate a CML report
9. Post the report to the pull request

## Stack

GCP, GCS, DVC, Feast, MLflow, scikit-learn, GitHub Actions, CML, pytest
