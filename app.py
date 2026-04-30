# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load model
pipe = pickle.load(open("pipe_logistic.pkl","rb"))

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    color: #ff4b4b;
    text-align: center;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🩺 Diabetes Prediction App")

st.markdown("### Enter Patient Details Below 👇")

# Create 2 columns
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("🤰 Pregnancies", min_value=0.0)
    Glucose = st.number_input("🍬 Glucose Level", min_value=0.0)
    BloodPressure = st.number_input("💓 Blood Pressure", min_value=0.0)
    SkinThickness = st.number_input("📏 Skin Thickness", min_value=0.0)

with col2:
    Insulin = st.number_input("💉 Insulin", min_value=0.0)
    BMI = st.number_input("⚖️ BMI", min_value=0.0)
    DiabetesPedigreeFunction = st.number_input("🧬 Diabetes Pedigree", min_value=0.0)
    Age = st.number_input("🎂 Age", min_value=0.0)

st.markdown("---")

# Predict button
if st.button("🔍 Predict Diabetes"):

    input_df = pd.DataFrame({
        "Pregnancies":[Pregnancies],
        "Glucose":[Glucose],
        "BloodPressure":[BloodPressure],
        "SkinThickness":[SkinThickness],
        "Insulin":[Insulin],
        "BMI":[BMI],
        "DiabetesPedigreeFunction":[DiabetesPedigreeFunction],
        "Age":[Age]
    })

    prediction = pipe.predict(input_df)

    # Output styling
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ No Diabetes Detected")

    st.subheader("📊 Input Summary")
    st.dataframe(input_df)