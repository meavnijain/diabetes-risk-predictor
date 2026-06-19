import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Behtar aur bada Diagnostic Dataset (Predictor math balanced)
data = {
    'Glucose': [85, 190, 90, 160, 110, 175, 95, 185, 105, 165, 75, 195, 80, 150, 115],
    'BloodPressure': [70, 90, 65, 85, 72, 88, 70, 95, 74, 80, 60, 100, 68, 82, 75],
    'BMI': [22.0, 38.5, 24.1, 32.6, 26.4, 34.1, 21.5, 40.2, 25.8, 30.7, 19.8, 42.1, 23.5, 31.2, 27.1],
    'Age': [23, 52, 28, 45, 31, 49, 24, 55, 35, 41, 22, 58, 26, 47, 33],
    'Outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 0 = Safe, 1 = Risk
}
df = pd.DataFrame(data)

X = df[['Glucose', 'BloodPressure', 'BMI', 'Age']]
y = df['Outcome']

# Train Model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# 4. Streamlit UI Settings
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🏥")
st.title("🏥 AI Diabetes Risk Predictor")
st.write("Change the sliders to see live parameter changes in the table below.")

st.sidebar.header("📋 Patient Diagnostic Metrics")

# Added unique 'key' to every slider so streamlit saves the state correctly
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 50, 200, 110, key="sb_glucose")
bp = st.sidebar.slider("Blood Pressure (mm Hg)", 40, 130, 75, key="sb_bp")
bmi = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0, key="sb_bmi")
age = st.sidebar.slider("Age (Years)", 1, 100, 30, key="sb_age")

# Live Input Monitor
user_data = pd.DataFrame([[glucose, bp, bmi, age]], columns=['Glucose', 'BloodPressure', 'BMI', 'Age'])
st.subheader("📊 Live Patient Parameters")
st.write(user_data) # Is table mein ab aapko slider badalte hi live numbers badalte dikhenge!

# Prediction Analysis
if st.button("Analyze Risk Profile"):
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)
    
    st.subheader("🩺 Diagnostic Evaluation Result")
    if prediction[0] == 1:
        st.error(f"🚨 High Risk Detected: The model estimates a {prediction_proba[0][1]*100:.1f}% probability of diabetes.")
    else:
        st.success(f"✅ Low Risk Detected: The model estimates a {prediction_proba[0][0]*100:.1f}% probability of a healthy profile.")
