import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

# ----------------------------------------------------------------------------
# Load the trained pipeline (model + scaler + feature order)
# rf_pipeline.pkl was saved as: {"model": rf_classifier, "scaler": scaler, "features": [...]}
# ----------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("rf_pipeline.pkl")
    return pipeline["model"], pipeline["scaler"], pipeline["features"]

model, scaler, feature_order = load_pipeline()

# Only these columns were scaled during training (see notebook, cell 44)
NUMERICAL_FEATURES = list(scaler.feature_names_in_)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "Enter a patient's clinical details below and the trained Random Forest "
    "model will estimate the likelihood of heart disease."
)

st.divider()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("patient_form"):
    st.subheader("Patient information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=54)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        chest_pain_type = st.selectbox(
            "Chest pain type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – Typical angina",
                1: "1 – Atypical angina",
                2: "2 – Non-anginal pain",
                3: "3 – Asymptomatic",
            }[x],
        )
        resting_blood_pressure = st.number_input(
            "Resting blood pressure (mm Hg)", min_value=80, max_value=220, value=130
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dl)", min_value=100, max_value=600, value=245
        )
        fasting_blood_sugar = st.selectbox(
            "Fasting blood sugar > 120 mg/dl?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )
        ecg = st.selectbox(
            "Resting ECG result",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 – Normal",
                1: "1 – ST-T wave abnormality",
                2: "2 – Left ventricular hypertrophy",
            }[x],
        )

    with col2:
        max_heart_rate = st.number_input(
            "Max heart rate achieved", min_value=60, max_value=220, value=150
        )
        exercise_induced_chest_pain = st.selectbox(
            "Exercise-induced angina?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )
        st_depression = st.number_input(
            "ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f"
        )
        st_slope = st.selectbox(
            "Slope of peak exercise ST segment",
            options=[0, 1, 2],
            format_func=lambda x: {0: "0 – Upsloping", 1: "1 – Flat", 2: "2 – Downsloping"}[x],
        )
        stained_blood_vessels = st.selectbox(
            "Number of major vessels stained (0–4)", options=[0, 1, 2, 3, 4]
        )
        blood_disorder = st.selectbox(
            "Blood disorder (thalassemia code)",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – Unknown",
                1: "1 – Fixed defect",
                2: "2 – Normal",
                3: "3 – Reversible defect",
            }[x],
        )

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    raw_input = {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "ecg": ecg,
        "max_heart_rate": max_heart_rate,
        "exercise_induced_chest_pain": exercise_induced_chest_pain,
        "st_depression": st_depression,
        "st_slope": st_slope,
        "stained_blood_vessels": stained_blood_vessels,
        "blood_disorder": blood_disorder,
    }

    # Build a single-row DataFrame in the exact column order the model expects
    input_df = pd.DataFrame([raw_input])[feature_order]

    # Scale only the numerical columns, exactly like in training
    scaled_df = input_df.copy()
    scaled_df[NUMERICAL_FEATURES] = scaler.transform(input_df[NUMERICAL_FEATURES])

    prediction = model.predict(scaled_df)[0]
    probability = model.predict_proba(scaled_df)[0][1]

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease — predicted probability: **{probability:.1%}**")
    else:
        st.success(f"✅ Low risk of heart disease — predicted probability: **{probability:.1%}**")

    st.progress(min(max(probability, 0.0), 1.0))

    with st.expander("See the exact values sent to the model"):
        st.dataframe(input_df)

st.divider()
st.caption(
    "This tool is trained on a public heart disease dataset for educational purposes only "
    "and is not a substitute for professional medical advice."
)