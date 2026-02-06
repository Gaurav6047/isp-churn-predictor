import streamlit as st
import joblib
import pandas as pd

model = joblib.load("churn_pipeline.pkl")

st.title("ISP Customer Churn Prediction System”")

st.write("Enter customer details")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=5)

monthly = st.number_input("Monthly Charges", value=70.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

support = st.selectbox("Tech Support", ["Yes", "No"])
security = st.selectbox("Online Security", ["Yes", "No"])

payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)

# ---- Encoding same as training ----
contract_map = {
    "Month-to-month":0,
    "One year":1,
    "Two year":2
}

internet_map = {
    "No":0,
    "DSL":1,
    "Fiber optic":2
}

payment_map = {
    "Electronic check":0,
    "Mailed check":1,
    "Bank transfer (automatic)":2,
    "Credit card (automatic)":3
}

if st.button("Predict Risk"):

    data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": tenure * monthly,
        "Contract": contract_map[contract],
        "InternetService": internet_map[internet],
        "PaymentMethod": payment_map[payment],
        "TechSupport": 1 if support=="Yes" else 0,
        "OnlineSecurity": 1 if security=="Yes" else 0
    }])

    prob = model.predict_proba(data)[0][1]

    if prob > 0.7:
        risk = "HIGH RISK"
        action = "Call + Discount + Tech Support Offer"
    elif prob > 0.4:
        risk = "MEDIUM RISK"
        action = "Send Offer SMS / Plan Upgrade"
    else:
        risk = "LOW RISK"
        action = "Normal Monitoring"

    st.subheader(f"Risk: {risk}")
    st.write("Probability:", round(prob,3))
    st.write("Suggested Action:", action)
