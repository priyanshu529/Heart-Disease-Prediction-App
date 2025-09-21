import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and expected columns
model = joblib.load("knn_heart.pkl")
scaler = joblib.load("scaler_heart.pkl")
expected_columns = joblib.load("columns_heart.pkl")

st.title("Heart Disease Prediction")
st.markdown("Provide the following details to check your heart disease risk:")

# Collect user input
age = st.slider("Age", 0, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    # Start with all-zero row for expected features
    input_df = pd.DataFrame([[0] * len(expected_columns)], columns=expected_columns)

    # Fill numeric features
    if "Age" in input_df.columns: input_df.at[0, "Age"] = age
    if "RestingBP" in input_df.columns: input_df.at[0, "RestingBP"] = resting_bp
    if "Cholesterol" in input_df.columns: input_df.at[0, "Cholesterol"] = cholesterol
    if "FastingBS" in input_df.columns: input_df.at[0, "FastingBS"] = fasting_bs
    if "MaxHR" in input_df.columns: input_df.at[0, "MaxHR"] = max_hr
    if "Oldpeak" in input_df.columns: input_df.at[0, "Oldpeak"] = oldpeak

    # Fill categorical one-hot encoded features (only if they exist)
    for feat in [
        "Sex_" + sex,
        "ChestPainType_" + chest_pain,
        "RestingECG_" + resting_ecg,
        "ExerciseAngina_" + exercise_angina,
        "ST_Slope_" + st_slope
    ]:
        if feat in input_df.columns:
            input_df.at[0, feat] = 1

    # Reorder columns exactly as expected
    input_df = input_df[expected_columns]

    # Scale & predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease ⚠️")
    else:
        st.success("✅ Low Risk of Heart Disease ✅")
