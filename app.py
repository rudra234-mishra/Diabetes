# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import time

# Load model
pipe = pickle.load(open("pipe_logistic.pkl","rb"))

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

# Sidebar
st.sidebar.title("🩺 About")
st.sidebar.info("""
This app predicts the likelihood of diabetes using a Machine Learning model.

👨‍💻 Developed by Rudra  
""")

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
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🩺 Diabetes Prediction App")
st.markdown("### Enter Patient Details 👇")

# Layout
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.slider("🤰 Pregnancies", 0, 20, 1)
    Glucose = st.slider("🍬 Glucose Level", 0, 200, 100)
    BloodPressure = st.slider("💓 Blood Pressure", 0, 150, 70)
    SkinThickness = st.slider("📏 Skin Thickness", 0, 100, 20)

with col2:
    Insulin = st.slider("💉 Insulin", 0, 900, 80)
    BMI = st.slider("⚖️ BMI", 0.0, 60.0, 25.0)
    DiabetesPedigreeFunction = st.slider("🧬 Diabetes Pedigree", 0.0, 2.5, 0.5)
    Age = st.slider("🎂 Age", 1, 100, 25)

st.markdown("---")

# Predict
if st.button("🔍 Predict Diabetes"):

    with st.spinner("Analyzing data..."):
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

        # Probability (if model supports it)
        try:
            prob = pipe.predict_proba(input_df)[0][1]
        except:
            prob = None

    # Result Display
    st.markdown("## 🧾 Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk (No Diabetes)")

    # Show probability
    if prob is not None:
        st.metric("📊 Risk Probability", f"{prob*100:.2f}%")

        st.progress(int(prob * 100))

    st.markdown("---")

    # Show input nicely
    st.subheader("📋 Input Summary")
    st.dataframe(input_df, use_container_width=True)

    # Tips
    st.markdown("### 💡 Health Tips")
    st.info("""
    ✔ Maintain healthy diet  
    ✔ Exercise regularly  
    ✔ Monitor blood sugar  
    ✔ Avoid excess sugar  
    """)

---

# ✨ What’s Improved

## 🎨 UI
- Sliders instead of boring inputs  
- Sidebar info  
- Clean layout  

## ⚡ Interaction
- Loading spinner  
- Risk percentage  
- Progress bar  

## 🧠 Smart Features
- Shows probability  
- Health suggestions  

---

# 💡 Next Level Ideas (if you want)

I can help you add:

✔ Graph (risk visualization)  
✔ Download report (PDF)  
✔ User login system  
✔ Database storage  
✔ Deploy with custom domain  

---

# 🚀 Final Result

```text
Your app now looks like a real healthcare AI product
