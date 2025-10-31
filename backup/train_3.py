"""
train.py

Train a (Logistic Regression / Random Forest) classification model,
optionally run hyperparameter tuning, and log parameters, metrics, and model to MLflow.
Also saves a local exported_model directory so the model can be served by an API.

Usage examples:
  python train.py --data-path loan_default_sample.csv --target target --tune --model-type logistic --tune
  python train.py --data-path loan_default_sample.csv --target target --model-type random_forest

"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split, GridSearchCV
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
from datetime import datetime, timezone


def parse_args():
    parser = argparse.ArgumentParser(description="Train classification model and log to MLflow")
    parser.add_argument("--data-path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, default="", help="Target column name. If empty, last column is used.")
    parser.add_argument("--model-type", type=str, choices=["logistic", "random forest"], default="logistic")
    parser.add_argument("--tune", action="store_true", help="Run grid search hyperparameter tuning")
    parser.add_argument("--cv", type=int, default=5, help="CV folds for GridSearch")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--experiment-name", type=str, default="Classification-Experiment")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                        help="MLflow tracking URI (e.g., http://127.0.0.1:9000)")
    parser.add_argument("--register-model", type=str, default=None, help="Register model under this name in MLflow")
    parser.add_argument("--autolog", action="store_true", help="Enable mlflow.sklearn.autolog()")
    return parser.parse_args()



def main():
    args = parse_args()

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    if args.autolog:
        mlflow.sklearn.autolog()

    # Load data
    df = pd.read_csv(args.data_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Identify target
    target_col = args.target if args.target else df.columns[-1]
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in data")

    # Drop missing target rows
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Treat term_months as categorical
    if "term_months" in X.columns:
        X["term_months"] = X["term_months"].astype(str)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Preprocessing
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

   # Choose estimator
    if args.model_type == "logistic":
        estimator = LogisticRegression(max_iter=1000, random_state=args.random_state)
        param_grid = {
            "estimator__C": [0.01, 0.1, 1, 10],
            "estimator__penalty": ["l2"],
            "estimator__solver": ["lbfgs", "liblinear"]
        }
    elif args.model_type == "random forest":
        estimator = RandomForestClassifier(random_state=args.random_state)
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [5, 10, 20, None],
            "estimator__min_samples_split": [2, 5, 10],
            "estimator__class_weight": ["balanced", None]
        }

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", estimator)
    ])

    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"train_{args.model_type}_{timestamp}") as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        if args.tune:
            print(f"Running hyperparameter tuning for {args.model_type}...")
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=args.cv,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=2
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            mlflow.log_params(search.best_params_)
            print("Best params:", search.best_params_)
        else:
            pipeline.fit(X_train, y_train)
            model = pipeline

        # Evaluate on test set
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Log metrics
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("roc_auc", float(auc))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("f1_score", float(f1))

        # -----------------------------
        # Save model
        # -----------------------------
        export_dir = os.path.abspath("exported_model")
        if os.path.exists(export_dir):
            import shutil
            shutil.rmtree(export_dir)

        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name=args.register_model if args.register_model else None
        )

        mlflow.sklearn.save_model(model, export_dir)
        print(f"Saved exported model to: {export_dir}")

        # Save metadata
        meta = {
            "run_id": run_id,
            "metrics": {
                "accuracy": accuracy,
                "roc_auc": auc,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            },
            "params": {"model_type": args.model_type}
        }
        with open("run_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Training complete. MLflow run id: {run_id}")

if __name__ == "__main__":
    main()
