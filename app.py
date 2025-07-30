import streamlit as st
import pandas as pd
import joblib


st.title("Salary Predictor")

work_year = st.number_input("Work Year", min_value=2000, max_value=2025, value=2024)
remote_ratio = st.slider("Remote Ratio", 0, 100, 50)
job_title = st.selectbox("Job Title", [
    "Data Scientist", "Data Engineer", "ML Engineer", "Software Engineer",
    "Research Scientist", "AI Scientist", "Analyst", "Data Analyst", "BI Analyst"
])

employee_residence = st.selectbox("Employee Residence", [
    "United States", "India", "Germany", "Canada", "United Kingdom",
    "France", "Australia", "Brazil", "Spain", "Netherlands"
])

company_location = st.selectbox("Company Location", [
    "United States", "India", "Germany", "Canada", "United Kingdom",
    "France", "Australia", "Brazil", "Spain", "Netherlands"
])

experience_level = st.selectbox("Experience Level", ["EN", "EX", "MI", "SE"])
employment_type = st.selectbox("Employment Type", ["FT", "PT", "CT", "FL"])
company_size = st.selectbox("Company Size", ["S", "M", "L"])

if st.button("Predict Salary"):
    input_df = pd.DataFrame([{
        "work_year": work_year,
        "remote_ratio": remote_ratio,
        "job_title": job_title,
        "employee_residence": employee_residence,
        "company_location": company_location,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "company_size": company_size,
    }])

    model = joblib.load("model.pkl")
    full_df = pd.read_csv("salaries.csv")
    full_df = pd.concat([full_df, input_df], ignore_index=True)
    processed = preprocess_data(full_df)
    pred_input = processed.iloc[-1:].drop(columns=["salary_in_usd"], errors="ignore")
    prediction = model.predict(pred_input)[0]
    st.success(f"Predicted Salary (USD): ${prediction:,.2f}")
