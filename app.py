import numpy as np
import pandas as pd
import pickle
import streamlit as st
import time
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_model():
    with open("pipe_logistic.pkl", "rb") as f:
        return pickle.load(f)

pipe = load_model()

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Settings")

preset = st.sidebar.selectbox(
    "Load Sample Data",
    ["Custom", "Healthy Person", "High Risk Person"]
)

# Presets
if preset == "Healthy Person":
    default = [1, 100, 70, 20, 80, 24.0, 0.4, 25]
elif preset == "High Risk Person":
    default = [6, 180, 95, 35, 250, 35.0, 1.2, 50]
else:
    default = [1, 100, 70, 20, 80, 25.0, 0.5, 25]

# Title
st.title("🩺 Diabetes Prediction App Made By Rudra")
st.markdown("### Enter Patient Details 👇")

# Layout
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.slider("🤰 Pregnancies", 0, 20, default[0])
    Glucose = st.slider("🍬 Glucose", 0, 200, default[1])
    BloodPressure = st.slider("💓 Blood Pressure", 0, 150, default[2])
    SkinThickness = st.slider("📏 Skin Thickness", 0, 100, default[3])

with col2:
    Insulin = st.slider("💉 Insulin", 0, 900, default[4])
    BMI = st.slider("⚖️ BMI", 0.0, 60.0, default[5])
    DiabetesPedigreeFunction = st.slider("🧬 Pedigree", 0.0, 2.5, default[6])
    Age = st.slider("🎂 Age", 1, 100, default[7])

st.markdown("---")

# Predict
if st.button("🚀 Predict Now"):

    with st.spinner("🔍 Analyzing..."):
        time.sleep(1)

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

        prediction = pipe.predict(input_df)[0]

        try:
            prob = pipe.predict_proba(input_df)[0][1]
        except:
            prob = None

    st.markdown("## 🧾 Prediction Result")

    # Result Card
    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    # Probability Gauge
    if prob is not None:
        st.metric("📊 Risk Probability", f"{prob*100:.2f}%")
        st.progress(int(prob * 100))

    st.markdown("---")

    # Visualization
    st.subheader("📊 Feature Overview")

    fig, ax = plt.subplots()
    features = input_df.columns
    values = input_df.iloc[0]

    ax.barh(features, values)
    ax.set_title("Input Feature Values")

    st.pyplot(fig)

    # Input Table
    st.subheader("📋 Input Summary")
    st.dataframe(input_df, use_container_width=True)

    # Smart Feedback
    st.subheader("🧠 AI Health Suggestions")

    if Glucose > 140:
        st.warning("⚠️ High glucose detected — consider reducing sugar intake.")

    if BMI > 30:
        st.warning("⚠️ High BMI — regular exercise is recommended.")

    if Age > 45:
        st.info("ℹ️ Regular health checkups are important.")

    st.success("✔ Stay healthy and monitor regularly!")
