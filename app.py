import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Pima Indians Diabetes Predictor", page_icon="🌲", layout="centered"
)

model = joblib.load("random_forest_model.joblib")
