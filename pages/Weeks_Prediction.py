import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and preprocessing tools
model = joblib.load("pages/model_1.pkl")
scaler = joblib.load("pages/scaler_1.pkl")  # Ensure the scaler was saved during training
label_encoders = joblib.load("pages/label_encoders_1.pkl")  # Ensure encoders were saved

# Define dropdown options
countries = ['Spain', 'UK', 'Switzerland', 'France', 'USA', 'Germany', 'Russia', 'Brazil', 'India', 'Australia', 'Canada', 'South Korea', 'South Africa', 'Mexico', 'China', 'Japan', 'Sweden', 'Singapore', 'Italy', 'Netherlands']
clinical_departments = ['Cardiology', 'Neurology', 'Oncology', 'Radiology', 'Endocrinology']
phases = ['I', 'II', 'III', 'IV']

def predict_ctt_fp_weeks(input_data):
    input_df = pd.DataFrame([input_data])  # Convert input to DataFrame

    # Encode categorical variables (ONLY for non-binary columns)
    categorical_cols = ["Country", "Clinical_Department", "Phase"]
    for col in categorical_cols:
        if col in label_encoders:
            input_df[col] = input_df[col].astype(str)  # Convert to string for safety
            if not set(input_df[col]).issubset(set(label_encoders[col].classes_)):  # Handle unseen labels
                st.error(f"Error: New category '{input_df[col].values[0]}' found in column '{col}'")
                return None
            input_df[col] = label_encoders[col].transform(input_df[col])  # Apply encoding

    # Scale numerical features
    input_df_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_df_scaled)[0]
    return round(prediction, 2)

# Streamlit UI
st.title("Clinical Trial Forecasting")

# User Inputs
country = st.selectbox("Select Country", countries)
clinical_department = st.selectbox("Select Clinical Department", clinical_departments)
phase = st.selectbox("Select Phase", phases)
rare_disease = st.selectbox("Rare Disease", ['Yes', 'No'])
new_indication = st.selectbox("New Indication", ['Yes', 'No'])
elderly = st.selectbox("Elderly", ['Yes', 'No'])
number_of_sites = st.number_input("Number of Sites", min_value=1, step=1)

# Convert categorical Yes/No to numerical 1/0 (DO NOT ENCODE AGAIN)
rare_disease = 1 if rare_disease == 'Yes' else 0
new_indication = 1 if new_indication == 'Yes' else 0
elderly = 1 if elderly == 'Yes' else 0

if st.button("Predict"):
    user_input = {
        "Clinical_Department": clinical_department,
        "Country": country,
        "Phase": phase,
        "Rare_Disease": rare_disease,  # No encoding needed
        "New_Indication": new_indication,  # No encoding needed
        "Elderly": elderly,  # No encoding needed
        "Number_of_Sites": number_of_sites
    }
    
    prediction = predict_ctt_fp_weeks(user_input)
    if prediction is not None:
        st.success(f"Predicted CTT FP Weeks: {prediction}")


col1, col2 = st.columns(2)
with col1:
    if st.button("Go to HOME"):
        st.switch_page("Home_Page.py")