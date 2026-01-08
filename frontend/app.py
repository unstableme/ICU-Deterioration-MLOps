import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(page_title="ICU Deterioration Simulator", layout="wide")
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
WINDOW_SIZE = 12
MAX_HOUR = 48
vitals_name = ['ALP', 'ALT', 'AST', 'Age', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol',
            'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Gender', 'Glucose', 'HCO3', 'HCT',
            'HR', 'Height', 'ICUType', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP',
            'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'RespRate', 'SaO2', 
            'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight', 'pH'
            ]
st.sidebar.title("ICU Deterioration Prediction Model")
st.sidebar.markdown("Select Patient")

def get_patient_ids():
    response = requests.get(f"{BACKEND_URL}/patients")
    return response.json()['patient_ids']
patient_ids = get_patient_ids()

selected_patient = st.sidebar.selectbox("Select Patient", patient_ids)
end_hour = st.sidebar.slider("Horizon(in hour):", WINDOW_SIZE, MAX_HOUR)

st.title(f"Patient {selected_patient} Overview")

def get_patients_vital(record_id, end_hour):
    response = requests.post(
        f"{BACKEND_URL}/predict",
        json={"record_id": str(record_id), "end_hour": end_hour}
    )
    return response.json()

result = get_patients_vital(selected_patient, end_hour)
# st.write("Backend response:", result)

vitals = result['vitals']
vitals_df = pd.DataFrame(vitals, columns=vitals_name)
st.subheader("Patients Vital Trends")
st.line_chart(vitals_df)

st.subheader("Risk Score")
risk_score = result['risk_score']
alert = result['alert']

st.metric("Risk Score", f"{risk_score:.2f}")


alert_colors = {
    "LOW": "#2ecc71",
    "MEDIUM": "#f39c12",
    "HIGH": "#e74c3c"
}

st.markdown(
    f"""
    <div style="
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        background-color: {alert_colors[alert]};
        color: white;
        font-size: 24px;
        font-weight: bold;">
        ALERT LEVEL: {alert}
    </div>
    """,
    unsafe_allow_html=True
)
