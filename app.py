import streamlit as st
import joblib
import pandas as pd
from sklearn.utils import _joblib as joblib  # Preferred for scikit-learn


# Load the model
model = joblib.load("depression_model.pkl")

# Feature list
features = ['Interest', 'Sleep', 'Fatigue', 'Appetite',
            'Worthlessness', 'Concentration', 'Agitation', 'Suicidal Ideation']

# Input sliders
st.title("Depression Detection App")

st.markdown("Please rate each symptom from 0 (None) to 3 (Severe):")

user_input = {}
for feature in features:
    user_input[feature] = st.slider(feature, 0, 3, 1)

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df).max() * 100

    label_map = {0: "Mild", 1: "Moderate", 2: "No depression", 3: "Severe"}
    result = label_map.get(prediction, "Unknown")

    st.success(f"Prediction: **{result}**")
    st.info(f"Model Confidence: **{confidence:.2f}%**")

