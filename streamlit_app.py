
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL AND DATASET
# -----------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/hospital_readmission_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/cleaned_data.csv",
        low_memory=False
    )


model = load_model()
df = load_data()

expected_columns = list(model.feature_names_in_)

# Columns treated as categorical even if stored as numbers
categorical_columns = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

# -----------------------------------
# HEADER
# -----------------------------------

st.title("🏥 Hospital Readmission Prediction")

st.markdown(
    "AI-powered system for estimating patient readmission risk"
)

st.divider()

st.subheader("Patient Information")

st.write(
    "Enter the patient's details below to generate a prediction."
)

# -----------------------------------
# INPUT FIELD FUNCTION
# -----------------------------------

def create_input(col):

    display_name = col.replace("_", " ").title()

    # Categorical fields
    if (
        col in categorical_columns
        or not pd.api.types.is_numeric_dtype(df[col])
    ):

        options = sorted(
            df[col].dropna().astype(str).unique().tolist()
        )

        if not options:
            options = ["Unknown"]

        value = st.selectbox(
            display_name,
            options=options,
            key=f"input_{col}"
        )

    # Numerical fields
    else:

        numeric_values = pd.to_numeric(
            df[col], errors="coerce"
        ).dropna()

        if numeric_values.empty:
            value = st.number_input(
                display_name,
                value=0.0,
                key=f"input_{col}"
            )

        else:
            minimum = float(numeric_values.min())
            maximum = float(numeric_values.max())
            default = float(numeric_values.median())

            if minimum == maximum:
                value = st.number_input(
                    display_name,
                    value=default,
                    key=f"input_{col}"
                )

            else:
                value = st.number_input(
                    display_name,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    key=f"input_{col}"
                )

            if pd.api.types.is_integer_dtype(df[col]):
                value = int(value)

    return value


# -----------------------------------
# GROUP PATIENT INFORMATION
# -----------------------------------

demographics = [
    "race",
    "gender",
    "age"
]

admission_details = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty"
]

medical_history = [
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses"
]

hospital_utilization = [
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient"
]

medication_details = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed"
]

groups = {
    "👤 Patient Demographics": demographics,
    "🏥 Hospital Admission Details": admission_details,
    "🩺 Medical History": medical_history,
    "📋 Hospital Stay & Utilization": hospital_utilization,
    "💊 Medication Details": medication_details
}

# Keep track of displayed fields
displayed_columns = set()

patient_data = {}

# -----------------------------------
# PATIENT INPUT FORM
# -----------------------------------

with st.form("patient_form"):

    for section, columns in groups.items():

        available_columns = [
            col for col in columns
            if col in expected_columns
        ]

        if not available_columns:
            continue

        st.markdown(f"### {section}")

        col1, col2 = st.columns(2)

        for index, col in enumerate(available_columns):

            with (col1 if index % 2 == 0 else col2):
                patient_data[col] = create_input(col)

            displayed_columns.add(col)

        st.divider()

    # Any remaining model features
    remaining_columns = [
        col for col in expected_columns
        if col not in displayed_columns
    ]

    if remaining_columns:

        st.markdown("### ⚙️ Additional Patient Details")

        col1, col2 = st.columns(2)

        for index, col in enumerate(remaining_columns):

            with (col1 if index % 2 == 0 else col2):
                patient_data[col] = create_input(col)

    st.divider()

    submit = st.form_submit_button(
        "🔍 Predict Readmission",
        use_container_width=True
    )

# -----------------------------------
# PREDICTION
# -----------------------------------

if submit:

    try:

        patient = pd.DataFrame([patient_data])

        # Ensure model receives columns in the correct order
        patient = patient[expected_columns]

        # Convert categorical ID fields to strings
        for col in categorical_columns:
            if col in patient.columns:
                patient[col] = patient[col].astype(str)

        prediction = int(model.predict(patient)[0])

        probability = float(
            model.predict_proba(patient)[0][1]
        )

        st.divider()

        st.subheader("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:
                st.error(
                    "⚠️ Readmission within 30 days predicted"
                )

            else:
                st.success(
                    "✅ No readmission within 30 days predicted"
                )

        with result_col2:

            st.metric(
                "Readmission Probability",
                f"{probability * 100:.2f}%"
            )

        st.markdown("### Risk Probability")

        st.progress(probability)

        st.caption(
            "This probability is a model estimate, not a clinical diagnosis."
        )

    except Exception as e:

        st.error(f"Prediction failed: {e}")

# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.caption(
    "Powered by Machine Learning | Hospital Readmission MLOps Project"
)

st.caption(
    "Educational prototype only — not intended for clinical decisions."
)
