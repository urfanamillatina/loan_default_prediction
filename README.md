# loan_default_prediction
This project focuses on building an automated loan default prediction system using machine learning techniques to assess the probability of default for each applicant, enabling proactive measures for risk mitigation.

## Contents
- `train.py` - Train a Logistic Regression and Random Forest model, log to MLflow, and export the model to `exported_model/`.
- `hyperparameter_search.py` - Example randomized search to tune Logistic Regression model and log best model to MLflow.
- `predict_api/` - FastAPI app and Dockerfile to serve `exported_model`.
- `MLproject`, `conda.yaml` - MLflow project metadata (optional).
- `requirements.txt` - Python dependencies for training.
- `predict_api/requirements.txt` - Dependencies for the API.
- `sample_input_for_regression.csv` - Example dataset (from the uploaded archive).

---


# Create a virtual environment and install dependencies
Using `venv`:
```bash
python -m venv .loanvenv
source .loanvenv/bin/activate
pip install -r requirements.txt
```

Or with conda:
```bash
conda env create -f conda.yaml
conda activate mlflow-linear-env
```

# MLFlow
```
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001
```

  Then open MLflow UI at: http://127.0.0.1:5001

  <p align="center">
  <img src="/Users/millatina/Documents/MU/Agentic AI/Assignments/Loan_Default_Prediction/loan_default_prediction/proof_screenshots/MLFlow.png" width="400"/><br/>
  <em>MLFlow</em>
</p>

# Train

## 1. LogisticRegression
```
python train.py --data-path loan_default_sample.csv --target target_default --model-type logistic --tune --mlflow-tracking-uri http://127.0.0.1:5001
```

## 2. RandomForest
```
python train.py --data-path loan_default_sample.csv --target target_default --model-type random_forest --tune --mlflow-tracking-uri http://127.0.0.1:5001
```

# Fine-tuning / Hyperparameter search

The model is tunned with GridSearch first, then tunned again with Randomizedsearch

1. **Grid search inside `train.py`** (use `--tune` with `--alpha ...`): uses `GridSearchCV` and logs the best params & model to MLflow.
2. **Randomized search example**: `python hyperparameter_search.py --data-path loan_default_sample.csv --target target_default --n-iter 20 --mlflow-tracking-uri http://127.0.0.1:5001`

<p align="center">
  <img src="/Users/millatina/Documents/MU/Agentic AI/Assignments/Loan_Default_Prediction/loan_default_prediction/proof_screenshots/tuning_random_search_logistic.png" width="400"/><br/>
  <em>Fine-tuning </em>
</p>


# Run the FastAPI app
Install API dependencies:
```bash
pip install -r predict_api/requirements.txt
```
Run:
```bash
#export MODEL_PATH=./exported_model
set MODEL_PATH=./exported_model
echo $MODEL_PATH
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
<p align="center">
  <img src="/Users/millatina/Documents/MU/Agentic AI/Assignments/Loan_Default_Prediction/loan_default_prediction/proof_screenshots/POST_api.png" width="400"/><br/>
  <em>Postman APi Testing</em>
</p>


# Docker
docker build -t ml-predict-api:latest -f predict_api/Dockerfile .

## For MacOS

### with MLFlow
```bash
docker run --rm -p 9000:9000 \
  -e MODEL_PATH=/app/exported_model \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 \
  -v "${PWD}/exported_model:/app/exported_model:ro" \
  --name ml-predict-api ml-predict-api:latest
```

### without MLFLOW

```bash
docker run --rm -p 9000:9000 \
  -e MODEL_PATH=/app/exported_model \
  -v "${PWD}\exported_model:/app/exported_model:ro" \
  --name ml-predict-api ml-predict-api:latest
```

## For Windows

### with MLflow

``` bash
docker run --rm -p 9000:9000 `
  -e MODEL_PATH=/app/exported_model `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 `
  -v "${PWD}\exported_model:/app/exported_model:ro" `
  --name ml-predict-api ml-predict-api:latest
```

### without MLFLOW

```bash
docker run --rm -p 9000:9000 `
  -e MODEL_PATH=/app/exported_model `
  -v "${PWD}\exported_model:/app/exported_model:ro" `
  --name ml-predict-api ml-predict-api:latest
```


## Streamlit
``` bash
streamlit run streamlit_app/app.py

```
<p align="center">
  <img src="/Users/millatina/Documents/MU/Agentic AI/Assignments/Loan_Default_Prediction/loan_default_prediction/proof_screenshots/streamlit.png" width="400"/><br/>
  <em>Streamlit User Interface </em>
</p>


# Dockerhub

## Build image
```
docker build -t mu55/ml-predict-api:latest -f predict_api/Dockerfile .
```

<p align="center">
  <img src="/Users/millatina/Documents/MU/Agentic AI/Assignments/Loan_Default_Prediction/loan_default_prediction/proof_screenshots/docker_image.png" width="400"/><br/>
  <em>Docker Image</em>
</p>


## Push the image to dockerhub
```
docker push mu55/ml-predict-api:latest
```
<p align="center">
  <img src="/Users/millatina/Documents/MU/Agentic AI/Assignments/Loan_Default_Prediction/loan_default_prediction/proof_screenshots/docker_hub.png" width="400"/><br/>
  <em>Docker Image on DockerHub</em>
</p>

## Pull from dockerhub from ubuntu (EC2 AWS)
```
docker pull mu55/ml-predict-api:latest
```

<p align="center">
  <img src="/Users/millatina/Documents/MU/Agentic AI/Assignments/Loan_Default_Prediction/loan_default_prediction/proof_screenshots/ubuntu_terminal_success.png" width="400"/><br/>
  <em>Ubuntu Terminal</em>
</p>

## Run the model
```
docker run --rm -p 9000:9000 -e MODEL_PATH=/app/exported_model --name ml-predict-api mu55/ml-predict-api:latest
```

<p align="center">
  <img src="/Users/millatina/Documents/MU/Agentic AI/Assignments/Loan_Default_Prediction/loan_default_prediction/proof_screenshots/docker_1.png" width="400"/><br/>
  <em>Run on Docker </em>
</p>

## POSTMAN

POST to `http://ec2-54-206-86-41.ap-southeast-2.compute.amazonaws.com:9000/predict` with payload:
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

<p align="center">
  <img src="/Users/millatina/Documents/MU/Agentic AI/Assignments/Loan_Default_Prediction/loan_default_prediction/proof_screenshots/EC2_postman_success.png" width="400"/><br/>
  <em>EC2 Postman - Success </em>
</p>
