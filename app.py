import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Pima Indians Diabetes Predictor", page_icon="🌲", layout="centered"
)


# 1. Load the pre-trained Random Forest model (cached for performance)
@st.cache_resource
def load_model():
    # Make sure 'pima_rf_model.joblib' is in the same directory as this script
    return joblib.load("random_forest_model.joblib")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "❌ Could not find 'random_forest_model.joblib'. Please place the model file in the same directory as this script."
    )
    st.stop()

# App Header
st.title("🌲 Pima Indians Diabetes Predictor")
st.markdown("Adjust the sliders in the sidebar to test real-time predictions.")

# 2. Setup the sidebar layout for inputs
st.sidebar.header("📊 Input Patient Features")

pregnancies = st.sidebar.slider(
    "Pregnancies", min_value=0, max_value=17, value=3, step=1
)
glucose = st.sidebar.slider(
    "Glucose Level", min_value=0, max_value=200, value=117, step=1
)
blood_pressure = st.sidebar.slider(
    "Blood Pressure (mm Hg)", min_value=0, max_value=122, value=72, step=1
)
skin_thickness = st.sidebar.slider(
    "Triceps Skin Fold Thickness (mm)", min_value=0, max_value=99, value=23, step=1
)
insulin = st.sidebar.slider(
    "2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=846, value=30, step=5
)
bmi = st.sidebar.slider(
    "Body Mass Index (BMI)", min_value=0.0, max_value=67.1, value=32.0, step=0.1
)
dpf = st.sidebar.slider(
    "Diabetes Pedigree Function",
    min_value=0.07,
    max_value=2.42,
    value=0.37,
    step=0.01,
)
age = st.sidebar.slider("Age (years)", min_value=21, max_value=81, value=29, step=1)

# 3. Map the exact feature names used during model training to avoid warnings
feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Gather current slider inputs
raw_features = [
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age,
]

# Convert to a DataFrame with valid column names
features = pd.DataFrame([raw_features], columns=feature_names)

# 4. Generate prediction and display metrics
prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]

st.subheader("🔮 Prediction Results")

# Display the outcome clearly using Streamlit alert boxes
if prediction == 1:
    risk_percentage = probabilities[1] * 100
    st.error(f"🚨 **Result: Positive for Diabetes**")
    st.metric(label="Calculated Risk Score", value=f"{risk_percentage:.1f}%")
else:
    confidence_percentage = probabilities[0] * 100
    st.success(f"🟢 **Result: Negative for Diabetes**")
    st.metric(
        label="Confidence Level", value=f"{confidence_percentage:.1f}%"
    )

# Visualizing the raw metrics array in the app UI
st.info(
    f"Probability Breakdowns — Negative: {probabilities[0]*100:.1f}% | Positive: {probabilities[1]*100:.1f}%"
)
