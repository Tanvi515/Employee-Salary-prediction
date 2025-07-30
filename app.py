import streamlit as st
import pandas as pd
import joblib

# Custom CSS for colors and layout
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        h1 {
            color: #4a90e2;
        }
        .stButton > button {
            background-color: #4a90e2;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Salary Predictor")
st.markdown("### 🧠 Enter details below and we'll predict your salary!")

# Input form with emojis
work_year = st.number_input("📅 Work Year", min_value=2000, max_value=2025, value=2024)
remote_ratio = st.slider("🌍 Remote Ratio (%)", 0, 100, 50)
job_title = st.text_input("👨‍💻 Job Title", "Data Scientist")
employee_residence = st.text_input("🏠 Employee Residence", "United States")
company_location = st.text_input("📍 Company Location", "United States")
experience_level = st.selectbox("🎓 Experience Level", ["EN", "EX", "MI", "SE"])
employment_type = st.selectbox("💼 Employment Type", ["FT", "PT", "CT", "FL"])
company_size = st.selectbox("🏢 Company Size", ["S", "M", "L"])

# Prediction button
if st.button("🚀 Predict Salary"):
    st.markdown("⏳ Running prediction, please wait...")
    
    # Create input DataFrame
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

    # Load model and process input
    model = joblib.load("model.pkl")
    full_df = pd.read_csv("salaries.csv")
    full_df = pd.concat([full_df, input_df], ignore_index=True)

    # You must define preprocess_data somewhere or import it
    processed = preprocess_data(full_df)
    pred_input = processed.iloc[-1:].drop(columns=["salary_in_usd"], errors="ignore")

    # Make prediction
    prediction = model.predict(pred_input)[0]

    # Display result with emojis
    if prediction < 50000:
        st.warning(f"💸 Predicted Salary: ${prediction:,.2f} — Consider upskilling!")
    elif prediction < 100000:
        st.info(f"💼 Predicted Salary: ${prediction:,.2f} — Looking solid!")
    else:
        st.success(f"🤑 Predicted Salary: ${prediction:,.2f} — Great job!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit", unsafe_allow_html=True)
