
import pandas as pd
import json

# Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Remove target and unnecessary columns
df = df.drop(
    columns=["encounter_id", "patient_nbr", "weight", "readmitted"],
    errors="ignore"
)

# Select one patient record
patient = df.iloc[0].to_dict()

# Convert missing values to None
patient = {
    key: None if pd.isna(value) else value
    for key, value in patient.items()
}

# Save as JSON
with open("test_patient.json", "w") as file:
    json.dump({"features": patient}, file, indent=4)

print("Test patient data generated successfully!")
print("Saved as test_patient.json")
