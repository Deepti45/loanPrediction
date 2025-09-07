
import streamlit as st
import pandas as pd
import pickle

# Load model
with open("loan_model.pkl", "rb") as f:
    model, feature_cols = pickle.load(f)

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")

st.title("🏦 Loan Approval Prediction App")
st.write("This app predicts whether a loan will be **Approved (Y)** or **Not Approved (N)**.")

# Sidebar inputs
st.sidebar.header("Applicant Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
app_income = st.sidebar.number_input("Applicant Income", min_value=0, step=100)
coapp_income = st.sidebar.number_input("Co-applicant Income", min_value=0, step=100)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=1)
loan_term = st.sidebar.selectbox("Loan Term (days)", [360, 120, 180, 240, 300, 480])
credit_history = st.sidebar.selectbox("Credit History", [0, 1])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert to dataframe (same order as training features)
input_dict = {
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": 3 if dependents == "3+" else int(dependents),
    "Education": 0 if education == "Graduate" else 1,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": app_income,
    "CoapplicantIncome": coapp_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
}

input_df = pd.DataFrame([input_dict])

# Predict
if st.sidebar.button("Predict Loan Approval"):
    prediction = model.predict(input_df[feature_cols])[0]
    result = "✅ Loan Approved (Y)" if prediction == 1 else "❌ Loan Not Approved (N)"
    st.success(f"Prediction: {result}")