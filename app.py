
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import joblib

# Create FastAPI application
app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="API for predicting hospital readmission risk",
    version="1.0"
)

# Load trained model
model = joblib.load("models/hospital_readmission_model.pkl")

# Get expected input columns
expected_columns = list(model.feature_names_in_)

# Input format
class PatientData(BaseModel):
    features: Dict[str, Any]


@app.get("/")
def home():
    return {
        "message": "Hospital Readmission Prediction API is running!"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: PatientData):

    try:
        patient = pd.DataFrame([data.features])

        # Check required columns
        missing = [
            col for col in expected_columns
            if col not in patient.columns
        ]

        extra = [
            col for col in patient.columns
            if col not in expected_columns
        ]

        if missing or extra:
            raise HTTPException(
                status_code=400,
                detail={
                    "missing_columns": missing,
                    "unexpected_columns": extra
                }
            )

        patient = patient[expected_columns]

        # Convert categorical ID columns
        for col in [
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id"
        ]:
            if col in patient.columns:
                patient[col] = patient[col].astype(str)

        prediction = int(model.predict(patient)[0])
        probability = float(model.predict_proba(patient)[0][1])

        return {
            "prediction": prediction,
            "result": (
                "Readmission within 30 days"
                if prediction == 1
                else "No readmission within 30 days predicted"
            ),
            "readmission_probability": round(probability, 4)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
