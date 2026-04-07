import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(page_title="Heart Failure Prediction", page_icon="🫀", layout="wide")

# 2. Load Assets
@st.cache_resource
def load_my_assets():
    model = load_model("heart_model.h5")
    scaler = joblib.load("scaler.pkl")
    data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    return model, scaler, data

try:
    model1, s_scaler, raw_data = load_my_assets()
except Exception as e:
    st.error("⚠️ Files Not Found! Ensure 'heart_model.h5' and 'scaler.pkl' are in the same folder.")
    st.stop()

# 3. Title
st.title("🫀 Heart Failure Mortality Prediction System")
st.markdown("Enter patient details in the sidebar and click the button below to see survival probability.")

# 4. Sidebar & Translators
yes_no_map = {"No": 0, "Yes": 1}
sex_map = {"Female": 0, "Male": 1}

st.sidebar.header("📋 Patient Clinical Metrics")

def get_user_input():
    age = st.sidebar.slider("Age", 40, 95, 60)
    cpk = st.sidebar.number_input("CPK (mcg/L)", value=582)
    ef = st.sidebar.slider("Ejection Fraction (%)", 14, 80, 38)
    platelets = st.sidebar.number_input("Platelets", value=263358.0)
    sc = st.sidebar.number_input("Serum Creatinine", value=1.1)
    ss = st.sidebar.number_input("Serum Sodium", value=137)
    time = st.sidebar.slider("Follow-up Period (Days)", 4, 285, 100)
    
    # Text-based inputs for the user
    anaemia_text = st.sidebar.selectbox("Anaemia", ["No", "Yes"])
    diabetes_text = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
    hbp_text = st.sidebar.selectbox("High Blood Pressure", ["No", "Yes"])
    smoking_text = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
    sex_text = st.sidebar.selectbox("Sex", ["Female", "Male"])

    features = {
        'age': age, 'anaemia': yes_no_map[anaemia_text], 'creatinine_phosphokinase': cpk,
        'diabetes': yes_no_map[diabetes_text], 'ejection_fraction': ef, 'high_blood_pressure': yes_no_map[hbp_text],
        'platelets': platelets, 'serum_creatinine': sc, 'serum_sodium': ss,
        'sex': sex_map[sex_text], 'smoking': yes_no_map[smoking_text], 'time': time
    }
    return pd.DataFrame(features, index=[0])

input_df = get_user_input()

st.subheader("📝 Current Patient Profile")
st.write(input_df)

# 5. Prediction Logic (This part shows the Percentage)
if st.button("🚀 Calculate Mortality Risk"):
    # Preprocessing
    scaled_input = s_scaler.transform(input_df)
    
    # Model Calculation
    prediction = model1.predict(scaled_input)
    mortality_prob = float(prediction[0][0])
    survival_prob = 1.0 - mortality_prob
    
    # --- PERCENTAGE DISPLAY SECTION ---
    st.subheader("🩺 Diagnostic Result")
    
    # Creating two big boxes for the percentages
    m1, m2 = st.columns(2)
    m1.metric(label="Survival Probability", value=f"{survival_prob:.1%}")
    m2.metric(label="Mortality Risk", value=f"{mortality_prob:.1%}")

    # Text Alert
    if mortality_prob > 0.5:
        st.error(f"High Risk: {mortality_prob:.2%} chance of mortality event.")
    else:
        st.success(f"Low Risk: {survival_prob:.2%} chance of survival.")

    # --- GRAPH SECTION ---
    st.subheader("📊 Clinical Analysis Graph")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=raw_data, x="ejection_fraction", y="serum_creatinine", 
                    hue="DEATH_EVENT", palette=['green', 'red'], alpha=0.5)
    plt.scatter(input_df['ejection_fraction'], input_df['serum_creatinine'], 
                color='blue', s=300, marker='X', label='Current Patient')
    plt.title("Heart Function (EF) vs Kidney Function (Creatinine)")
    plt.legend()
    st.pyplot(fig)
    