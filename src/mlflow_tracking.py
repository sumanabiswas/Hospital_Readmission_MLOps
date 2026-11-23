
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Set experiment name
mlflow.set_experiment("Hospital_Readmission_Prediction")

# Load saved model
model = joblib.load("models/hospital_readmission_model.pkl")

# Load test dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Repeat the same feature preparation used in Stage 2
df = df.drop(
    columns=["encounter_id", "patient_nbr", "weight"],
    errors="ignore"
)

X = df.drop("readmitted", axis=1)
y = df["readmitted"].astype(int)

categorical_ids = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

for col in categorical_ids:
    if col in X.columns:
        X[col] = X[col].astype(str)

# Use a fixed test split
from sklearn.model_selection import train_test_split

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(y_test, predictions, zero_division=0),
    "recall": recall_score(y_test, predictions, zero_division=0),
    "f1_score": f1_score(y_test, predictions, zero_division=0),
    "roc_auc": roc_auc_score(y_test, probabilities)
}

# Track experiment
with mlflow.start_run(run_name="Selected_Model"):

    mlflow.log_param("model_type", type(model.named_steps["classifier"]).__name__)
    mlflow.log_param("dataset", "Diabetes 130-US Hospitals")

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
    model,
    name="hospital_model",
    skops_trusted_types=["numpy.dtype"]
)
    print("MLflow experiment recorded!")
    print("Metrics:", metrics)
