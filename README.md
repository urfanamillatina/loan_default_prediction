# loan_default_prediction
This project focuses on building an automated loan default prediction system using machine learning techniques to assess the probability of default for each applicant, enabling proactive measures for risk mitigation.

# MLFlow
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001

  Then open MLflow UI at: http://127.0.0.1:5001

# Train

## 1. LogisticRegression
`python train.py --data-path loan_default_sample.csv --target target_default --model-type logistic --tune --mlflow-tracking-uri http://127.0.0.1:5001`

## 2. RandomForest
`python train.py --data-path loan_default_sample.csv --target target_default --model-type random_forest --tune --mlflow-tracking-uri http://127.0.0.1:5001`

# Run the FastAPI app
Install API dependencies:
```bash
pip install -r predict_api/requirements.txt
```
Run:
```bash
#export MODEL_PATH=./exported_model
set MODEL_PATH=./exported_model
echo %MODEL_PATH% 
uvicorn predict_api.app:app --host 0.0.0.0 --port 9000
```
POST to `http://127.0.0.1:9000/predict` with payload:
```json
{
    "records": [
        {
            "age": 32, 
            "annual_income": 60000, 
            "employment_length": 3, 
            "home_ownership": "RENT",
            "purpose": "credit_card", 
            "loan_amount": 15000, 
            "term_months": 36, 
            "interest_rate": 12.5, 
            "dti": 20.3, 
            "credit_score": 720, 
            "delinquency_2yrs": 0, 
            "num_open_acc": 6
        }
    ]
} 
```
Response:
```json
{
    "predictions": [
        {
            "prediction": 0,
            "label" : "Non-default",
            "probability of default": 0.07162073804200371
        }
    ]
}
```
## Fine-tuning / Hyperparameter search

The model is tunned with GridSearch first, then tunned again with Randomizedsearch

1. **Grid search inside `train.py`** (use `--tune` with `--alpha ...`): uses `GridSearchCV` and logs the best params & model to MLflow.
2. **Randomized search example**: ``python hyperparameter_search.py --data-path loan_default_sample.csv --target target_default --n-iter 20 --mlflow-tracking-uri http://127.0.0.1:5001


---

# Docker
docker build -t ml-predict-api:latest -f predict_api/Dockerfile .

(MacOS)

docker run --rm -p 9000:9000 \
  -e MODEL_PATH=/app/exported_model \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 \
  -v "${PWD}/exported_model:/app/exported_model:ro" \
  --name ml-predict-api ml-predict-api:latest
