import streamlit as st
import joblib
import numpy as np

st.title("Stroke Prediction Form")

model = joblib.load("model/stroke_model.pkl")

# Form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Encoding
def encode_inputs():
    data = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'gender_Male': 1 if gender == "Male" else 0,
        'ever_married_Yes': 1 if ever_married == "Yes" else 0,
        'work_type_Private': 1 if work_type == "Private" else 0,
        'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
        'work_type_children': 1 if work_type == "children" else 0,
        'work_type_Govt_job': 1 if work_type == "Govt_job" else 0,
        'Residence_type_Urban': 1 if Residence_type == "Urban" else 0,
        'smoking_status_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
        'smoking_status_never smoked': 1 if smoking_status == "never smoked" else 0,
        'smoking_status_smokes': 1 if smoking_status == "smokes" else 0
    }

    cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
            'gender_Male', 'ever_married_Yes', 'work_type_Govt_job',
            'work_type_Private', 'work_type_Self-employed', 'work_type_children',
            'Residence_type_Urban', 'smoking_status_formerly smoked',
            'smoking_status_never smoked', 'smoking_status_smokes']

    return np.array([data.get(col, 0) for col in cols]).reshape(1, -1)

if st.button("Predict"):
    features = encode_inputs()
    prediction = model.predict(features)[0]
    st.success(f"Prediction: {'Stroke' if prediction == 1 else 'No Stroke'}")
