import streamlit as st
import joblib
import pandas as pd

model = joblib.load("churn_pipeline.pkl")

# ===== SIDEBAR INFO =====
st.sidebar.title("About Project")

st.sidebar.write("""
**ISP Customer Churn Prediction**

• Dataset: 7k+ ISP customers  
• Model: Logistic Regression  
• Focus Metric: Recall (85%)  

**Key Insights**
- Month-to-month users most risky  
- No Tech Support → high churn  
- High monthly bill increases risk  
""")



# ===== MAIN PAGE =====

st.title("ISP Customer Churn Prediction System")

st.write("""
### Business Goal
Identify customers likely to leave so company can take actions:
- Discount offers  
- Tech support calls  
- Plan upgrade  
""")

# ===== INPUT FORM =====

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", 0, 72, 5)
    monthly = st.number_input("Monthly Charges", 0.0, 150.0, 70.0)

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
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

# ===== ENCODING =====

contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
internet_map = {"No":0, "DSL":1, "Fiber optic":2}
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

    st.subheader("Result")

    if prob > 0.7:
        risk = "HIGH RISK"
        color = "red"
        action = """
        • Immediate call  
        • Discount offer  
        • Free tech support  
        """
    elif prob > 0.4:
        risk = "MEDIUM RISK"
        color = "orange"
        action = "• Send offer SMS • Plan suggestion"
    else:
        risk = "LOW RISK"
        color = "green"
        action = "• Normal monitoring"

    st.markdown(f"### Risk Level: :{color}[{risk}]")
    st.write("Probability:", round(prob, 3))

    st.subheader("Recommended Action")
    st.write(action)

# ===== INTERVIEW SECTION =====

st.markdown("---")
st.subheader("Model Summary")

st.write("""
• Baseline Recall: 51%  
• After Class Weight: 85%  
• Key Drivers:  
  - Contract Type  
  - Tech Support  
  - Monthly Charges  

• Business Impact:  
  - Capture risky users early  
  - Positive ROI despite false alarms  
""")

