import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="Loan Default Prediction UI", layout="wide")

st.title("Loan Default Prediction - Streamlit UI")
st.markdown("Interactively prepare input records and call the model API `/predict`. Configure the API URL below.")

# --- API URL input ---
api_url = st.text_input("Model API URL", value="http://localhost:9000/predict")

# --- Single record input form ---
st.markdown("### Single record input")
with st.form("single_form"):
    age = st.number_input('age', value=30, min_value=18, max_value=100)
    annual_income = st.number_input('annual_income', value=50000.0)
    employment_length = st.number_input('employment_length (years)', value=5)
    home_ownership = st.selectbox('home_ownership', options=["OWN", "RENT", "MORTGAGE", "OTHER"])
    purpose = st.selectbox('purpose', options=[
        "debt_consolidation", "credit_card", "home_improvement", "other"])
    loan_amount = st.number_input('loan_amount', value=10000.0)
    term_months = st.number_input('term_months', value=36)
    interest_rate = st.number_input('interest_rate (%)', value=10.5)
    dti = st.number_input('dti (debt-to-income ratio)', value=15.0)
    credit_score = st.number_input('credit_score', value=700.0)
    delinquency_2yrs = st.number_input('delinquency_2yrs', value=0)
    num_open_acc = st.number_input('num_open_acc', value=5)

    submitted = st.form_submit_button("Predict single record")

    if submitted:
        record = {}
        record["age"]= age
        record["annual_income"]= annual_income
        record["employment_length"]= employment_length
        record["home_ownership"]= home_ownership
        record["purpose"]= purpose
        record["loan_amount"]= loan_amount
        record["term_months"]= term_months
        record["interest_rate"]= interest_rate
        record["dti"]= dti
        record["credit_score"] = credit_score
        record["delinquency_2yrs"]= delinquency_2yrs
        record["num_open_acc"]= num_open_acc
        
    
        payload = {"records": [record]}

        try:
            resp = requests.post(api_url, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            st.json(result)  # For debugging visibility

            # Extract prediction regardless of structure
            pred_data = result.get("predictions", None)

            if pred_data is not None:
                # Handle dict-like {"0": 0} or list-like [0]
                if isinstance(pred_data, dict):
                    pred_value = list(pred_data.values())[0]
                elif isinstance(pred_data, list):
                    pred_value = pred_data[0]
                else:
                    pred_value = None

                if pred_value is not None:
                    label = "Default" if int(pred_value) == 1 else "Non-default"
                    st.success(f"Prediction successful — The applicant is predicted to be: **{label}**")
                else:
                    st.warning("Could not interpret prediction value.")
            else:
                st.warning(" 'predictions' field missing from API response.")

        except Exception as e:
            st.error(f"Request failed: {e}")

# --- Batch input upload ---
st.markdown("---")
st.markdown("### Batch input (upload CSV)")
uploaded = st.file_uploader("Upload CSV with same columns as training (exclude target_default)", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("Predict batch"):
            payload = {"records": df.to_dict(orient="records")}
            try:
                resp = requests.post(api_url, json=payload, timeout=30)
                resp.raise_for_status()
                res = resp.json()

                preds = res.get("predictions", None)
                if preds is not None:
                    if isinstance(preds, dict):
                        preds = list(preds.values())
                    df['prediction'] = preds
                    df['prediction_label'] = df['prediction'].map({0: "Non-default", 1: "Default"})
                    st.success("Batch prediction successful")
                    st.dataframe(df)
                else:
                    st.warning("Unexpected response structure.")
                    st.write(res)
            except Exception as e:
                st.error(f"Batch request failed: {e}")

    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")

st.markdown("---")
st.markdown("### Quick tips")
st.markdown("- Ensure the API URL points to your model server’s `/predict` endpoint.")
st.markdown("- The uploaded CSV must have identical feature columns as your training data (exclude target).")
st.markdown("- Example API URL if using Docker locally: `http://localhost:9000/predict`")

# Run this app using:
# streamlit run streamlit_app/app.py
