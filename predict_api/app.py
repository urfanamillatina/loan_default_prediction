# predict_api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import mlflow.sklearn
import os

app = FastAPI(title="Loan Default Prediction API", version="1.0")

# --- Load model ---
MODEL_PATH = os.getenv("MODEL_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "../exported_model")))

try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    print(f"Model failed to load: {e}")

# --- Define schemas ---
class LoanData(BaseModel):
    age: int
    annual_income: float
    employment_length: int
    home_ownership: str
    purpose: str
    loan_amount: float
    term_months: int
    interest_rate: float
    dti: float
    credit_score: int
    delinquency_2yrs: int
    num_open_acc: int

class LoanRequest(BaseModel):
    records: List[LoanData]


# --- Routes ---
@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify if the model is loaded successfully.
    """
    if model_loaded:
        return {"status": "ok", "model_loaded": True, "model_path": MODEL_PATH}
    else:
        return {"status": "error", "model_loaded": False, "message": "Model failed to load"}


@app.post("/predict")
def predict(data: LoanRequest):
    try:
        if not model_loaded or model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Convert list of LoanData → DataFrame
        df = pd.DataFrame([record.dict() for record in data.records])
        predictions = model.predict(df).tolist()
        probabilities = model.predict_proba(df)[:, 1].tolist()

        results = []
        for pred, prob in zip(predictions, probabilities):
            label = "Default" if pred == 1 else "Non-default"
            results.append({
                "prediction": int(pred),
                "label": label,
                "probability_of_default": float(prob)
            })

        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
