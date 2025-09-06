import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load('pages/fsi_prediction_model_test.pkl')
scaler = joblib.load('pages/scaler_test.pkl')
df = pd.read_csv('pages/synthetic_clinical_trial_data.csv')  # Load dataset

# Extract the expected feature names from scaler
expected_features = scaler.feature_names_in_

# Clinical department categories seen during training
clinical_dept_categories = [
    "Clinical_Department_Endocrinology", "Clinical_Department_Infectious Disease",
    "Clinical_Department_Neurology", "Clinical_Department_Oncology"
]

# Streamlit UI
st.title("Clinical Trials FSI Demand Prediction")
st.write("Enter the trial details to predict FSI demand and find the top and least suitable countries.")

# User input fields
phase = st.selectbox("Phase", [1, 2, 3, 4])
rare_disease = st.radio("Rare Disease", [0, 1])
new_indication = st.radio("New Indication", [0, 1])
elderly_usage = st.radio("Used for Elderly", [0, 1])
clinical_department = st.selectbox(
    "Clinical Department",
    ["Endocrinology", "Infectious Disease", "Neurology", "Oncology"]
)

if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame([{
        'Phase': phase, 'Rare_Disease': rare_disease, 'New_Indication': new_indication,
        'Elderly_Usage': elderly_usage
    }])

    # One-hot encoding for 'Clinical_Department'
    for category in clinical_dept_categories:
        input_data[category] = 1 if category.endswith(clinical_department) else 0

    # Ensure all expected columns exist
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns

    # Drop any extra columns not in expected features
    input_data = input_data[expected_features]

    # Convert to NumPy and scale
    scaled_input = scaler.transform(input_data.to_numpy())

    # Predict FSI demand
    predicted_fsi_demand = model.predict(scaled_input)[0]

    # Filter dataset based on input parameters
    df_filtered = df[
        (df['Phase'] == phase) & (df['Rare_Disease'] == rare_disease) &
        (df['New_Indication'] == new_indication) & (df['Elderly_Usage'] == elderly_usage)
    ].copy()

    if df_filtered.empty:
        st.warning("No exact historical match found. Using full dataset for prediction.")
        df_filtered = df.copy()

    # Keep 'Country' column for grouping later
    df_filtered_countries = df_filtered[['Country']].copy()

    # One-hot encode clinical department for dataset filtering
    for category in clinical_dept_categories:
        df_filtered[category] = (df_filtered["Clinical_Department"] == category.split("_")[-1]).astype(int)

    # Drop non-numeric and unnecessary columns before prediction
    df_filtered = df_filtered.drop(columns=['Study_Code', 'FSI_Demand', 'Clinical_Department'])

    # Ensure feature alignment
    df_filtered = df_filtered.reindex(columns=expected_features, fill_value=0)

    # Predict FSI demand for all matching countries
    df_filtered['Predicted_FSI_Demand'] = model.predict(scaler.transform(df_filtered))

    # Add back 'Country' column after prediction
    df_filtered['Country'] = df_filtered_countries['Country'].values

    # Get top and least 5 countries
    top_5_countries = df_filtered.groupby('Country')['Predicted_FSI_Demand'].mean().nlargest(5).index.tolist()
    least_5_countries = df_filtered.groupby('Country')['Predicted_FSI_Demand'].mean().nsmallest(5).index.tolist()

    # Display results
    st.success(f"Predicted FSI Demand: {predicted_fsi_demand:.2f}")

    st.subheader("Top 5 Suitable Countries:")
    for country in top_5_countries:
        st.write(f"✅ {country}")

    st.subheader("Least 5 Suitable Countries:")
    for country in least_5_countries:
        st.write(f"❌ {country}")

col1, col2 = st.columns(2)
with col1:
    if st.button("Go to HOME"):
        st.switch_page("Home_Page.py")