import streamlit as st
import requests
st.title(" Customer Monthly Charge Predictor")
with st.form("customer_form"):
    age = st.number_input("Enter Age", min_value=0, max_value=100, step=1)
    tenure = st.number_input("Enter Tenure (in years)", min_value=0, step=1)

    services = st.multiselect(
        "Select Subscribed Services",
        options=["Internet", "Phone", "TV"],
        default=["Internet"]
    )

    submit = st.form_submit_button("Predict Monthly Charges")

if submit:
    input_data = {
        "age": age,
        "tenure": tenure,
        "services": services
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict/", json=input_data)

        if response.status_code == 200:
            predicted_charge = response.json()["predicted_monthly_charge"]
            st.success(f"Predicted Monthly Charge: ₹{predicted_charge}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"⚠ Failed to connect to FastAPI server.\n{e}")