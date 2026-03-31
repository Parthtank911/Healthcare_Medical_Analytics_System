import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.feature_eng import feature_engineering


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Healthcare Medical Analytics",
    layout="wide"
)

st.title("🏥 Healthcare & Medical Analytics System")


# ===============================
# LOAD ML MODEL
# ===============================
MODEL_PATH = "models/ml_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("ML model not found. Train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)


# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Disease Prediction",
        "Feature Importance",
        "ECG Deep Learning",
        "Medical Report"
    ]
)


# ===============================
# USER INPUT FORM
# ===============================
def user_input_form():
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", 1, 100, 45)
        BloodPressure = st.number_input("Blood Pressure", 80, 200, 120)
        Cholesterol = st.number_input("Cholesterol", 100, 400, 200)
        HeartRate = st.number_input("Heart Rate", 50, 200, 75)
        BMI = st.number_input("BMI", 10.0, 50.0, 24.0)

    with col2:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        SmokingStatus = st.selectbox("Smoking", ["No", "Yes"])
        Diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        PhysicalActivity = st.slider("Physical Activity (hrs/week)", 0, 10, 3)
        StressLevel = st.selectbox("Stress Level", ["Low", "High"])
        FamilyHistory = st.selectbox("Family History", ["No", "Yes"])
        Medication = st.selectbox("Medication", ["No", "Yes"])
        BloodSugar = st.number_input("Blood Sugar", 70, 300, 110)

    data = {
        "PatientID": 1,
        "Age": Age,
        "Gender": 1 if Gender == "Female" else 0,
        "BloodPressure": BloodPressure,
        "Cholesterol": Cholesterol,
        "HeartRate": HeartRate,
        "SmokingStatus": 1 if SmokingStatus == "Yes" else 0,
        "Diabetes": 1 if Diabetes == "Yes" else 0,
        "BMI": BMI,
        "PhysicalActivity": PhysicalActivity,
        "FamilyHistory": 1 if FamilyHistory == "Yes" else 0,
        "Medication": 1 if Medication == "Yes" else 0,
        "StressLevel": 1 if StressLevel == "High" else 0,
        "BloodSugar": BloodSugar
    }

    return pd.DataFrame([data])


# ===============================
# PDF REPORT GENERATOR
# ===============================
def generate_pdf(patient_data, prediction, probability):
    file_path = "medical_report.pdf"
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Medical Risk Assessment Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, f"Generated on: {datetime.now()}")

    y = height - 120
    c.setFont("Helvetica", 10)

    for key, value in patient_data.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 14

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 20, "Prediction Result")

    result = "HIGH RISK" if prediction == 1 else "LOW RISK"
    c.drawString(50, y - 40, f"Heart Disease Risk: {result}")
    c.drawString(50, y - 60, f"Probability: {probability:.2%}")

    c.save()
    return file_path


# ===============================
# PAGE 1: DISEASE PREDICTION
# ===============================
if page == "Disease Prediction":
    st.header("🧠 Heart Disease Risk Prediction")

    input_df = user_input_form()

    os.makedirs("temp", exist_ok=True)
    input_path = "temp/input.csv"
    output_path = "temp/engineered.csv"

    input_df.to_csv(input_path, index=False)
    feature_engineering(input_path, output_path)

    engineered_df = pd.read_csv(output_path)

    engineered_df = engineered_df[model.feature_names_in_]

    prediction = model.predict(engineered_df)[0]
    probability = model.predict_proba(engineered_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK of Heart Disease ({probability:.2%})")
    else:
        st.success(f"✅ LOW RISK of Heart Disease ({probability:.2%})")


# ===============================
# PAGE 2: FEATURE IMPORTANCE
# ===============================
elif page == "Feature Importance":
    st.header("📊 Feature Importance")

    fi_df = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(fi_df)

    fig, ax = plt.subplots()
    ax.barh(fi_df["Feature"], fi_df["Importance"])
    ax.invert_yaxis()
    ax.set_title("ML Feature Importance")
    st.pyplot(fig)


# ===============================
# PAGE 3: ECG DEEP LEARNING
# ===============================
elif page == "ECG Deep Learning":
    st.header("❤️ ECG Deep Learning Analysis")

    ecg_path = "data/predictions/dl_ecg_predictions.csv"

    if not os.path.exists(ecg_path):
        st.error("ECG prediction file not found.")
    else:
        ecg_preds = pd.read_csv(ecg_path)

        st.subheader("ECG Predictions Sample")
        st.dataframe(ecg_preds.head(20))

        # 🔑 Auto-detect prediction column
        pred_col = None
        for col in ecg_preds.columns:
            if "pred" in col.lower():
                pred_col = col
                break

        if pred_col is None:
            st.error("No prediction column found in ECG file.")
        else:
            counts = ecg_preds[pred_col].value_counts()

            fig, ax = plt.subplots()
            ax.bar(counts.index.astype(str), counts.values)
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Count")
            ax.set_title("ECG Risk Distribution")
            st.pyplot(fig)


# ===============================
# PAGE 4: MEDICAL REPORT
# ===============================
elif page == "Medical Report":
    st.header("📄 Medical Report")

    input_df = user_input_form()

    os.makedirs("temp", exist_ok=True)
    input_path = "temp/input.csv"
    output_path = "temp/engineered.csv"

    input_df.to_csv(input_path, index=False)
    feature_engineering(input_path, output_path)

    engineered_df = pd.read_csv(output_path)
    engineered_df = engineered_df[model.feature_names_in_]

    prediction = model.predict(engineered_df)[0]
    probability = model.predict_proba(engineered_df)[0][1]

    if st.button("Generate PDF Report"):
        pdf_path = generate_pdf(
            input_df.iloc[0].to_dict(),
            prediction,
            probability
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Medical Report",
                data=f,
                file_name="medical_report.pdf",
                mime="application/pdf"
            )
