"""
train.py

Train a Logistic Regression or Random Forest model, optionally with hyperparameter tuning,
and log parameters, metrics, and model to MLflow.

Usage examples:
  python train.py --data-path loan_default_sample.csv --target target_default --model-type logistic --tune
  python train.py --data-path loan_default_sample.csv --target target_default --model-type random_forest --tune
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
)
import mlflow
import mlflow.sklearn


def parse_args():
    parser = argparse.ArgumentParser(description="Train classification model and log to MLflow")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, default="", help="Target column name. If empty, last column is used.")
    parser.add_argument("--model-type", type=str, choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--tune", action="store_true", help="Whether to run hyperparameter tuning")
    parser.add_argument("--alpha", type=float, nargs="*", default=[0.1,1.0,10.0], help="Alpha values for the tuning")
    parser.add_argument("--cv", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="Classification-Experiment")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="Optional MLflow tracking URI")
    parser.add_argument("--register-model", type=str, default=None, help="Optional MLflow model registry name")
    parser.add_argument("--autolog", action="store_true", help="Enable mlflow.sklearn.autolog()")
    return parser.parse_args()



def main():
    args = parse_args()

    # Configure MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    if args.autolog:
        mlflow.sklearn.autolog()

    # Load dataset
    df = pd.read_csv(args.data_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    target_col = args.target if args.target else df.columns[-1]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col, 'loan_id'])
    y = df[target_col]

    # Ensure 'term_months' is categorical
    if "term_months" in X.columns:
        X["term_months"] = X["term_months"].astype(str)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Identify feature types
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Base estimator
    if args.model_type == "logistic":
        estimator = LogisticRegression(max_iter=1000, random_state=args.random_state)
    elif args.model_type == "random_forest":
        estimator = RandomForestClassifier(random_state=args.random_state)
    else:
        raise ValueError("Unsupported model type")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("estimator", estimator)
    ])

    # Hyperparameter search space
    if args.model_type == "logistic":
        param_grid = {
            "estimator__C": np.logspace(-3, 3, 20),
            "estimator__solver": ["lbfgs", "liblinear"],
            "estimator__penalty": ["l2"]
        }
    else:  # Random Forest
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [5, 10, 20, None],
            "estimator__min_samples_split": [2, 5, 10],
            "estimator__min_samples_leaf": [1, 2, 4],
            "estimator__bootstrap": [True, False]
        }
        

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"train_{args.model_type}_{timestamp}") as run:
        run_id = run.info.run_id

        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        
        

        # Train or tune
        if args.tune and param_grid:
            print("Starting grid search with params:", param_grid)
            search = GridSearchCV(pipeline, param_grid=param_grid, cv=args.cv, scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            mlflow.log_params(search.best_params_)
        else:
            best_model = pipeline.fit(X_train, y_train)
            best_params = {}

        # Evaluate
        preds = best_model.predict(X_test)
        probs = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds)
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Save model
        mlflow.sklearn.log_model(best_model, artifact_path="model",
                                 registered_model_name=args.register_model)

        export_dir = os.path.abspath("exported_model")
        if os.path.exists(export_dir):
            import shutil
            shutil.rmtree(export_dir)
        mlflow.sklearn.save_model(best_model, export_dir)
        print(f"✅ Model exported to: {export_dir}")

        # Save metadata
        meta = {
            "run_id": run_id,
            "model_type": args.model_type,
            "best_params": best_params,
            "metrics": metrics
        }
        with open("run_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print("\nTraining complete ✅")
        print("MLflow run ID:", run_id)
        print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
