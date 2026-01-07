import streamlit as st
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
WINDOW_SIZE = 12
MAX_HOUR = 48
np.random.seed(42)

# ---------------- MOCK DATA ----------------
patient_ids = [101, 102, 103]

def generate_fake_vitals(end_hour, window_size=12):
    """Generate realistic ICU vital trends"""
    hours = list(range(end_hour - window_size + 1, end_hour + 1))

    vitals = {
        "hour": hours,
        "heart_rate": np.random.normal(90, 8, window_size).clip(60, 140),
        "spo2": np.random.normal(96, 1.5, window_size).clip(85, 100),
        "resp_rate": np.random.normal(20, 3, window_size).clip(10, 40),
        "mean_bp": np.random.normal(75, 10, window_size).clip(40, 120),
    }

    return pd.DataFrame(vitals).set_index("hour")

def simulate_risk_score(vitals_df):
    """Simple heuristic risk simulation"""
    hr_risk = (vitals_df["heart_rate"].mean() - 80) / 40
    spo2_risk = (95 - vitals_df["spo2"].mean()) / 10
    bp_risk = (70 - vitals_df["mean_bp"].mean()) / 30

    risk = np.clip(0.5 * hr_risk + 0.7 * spo2_risk + 0.6 * bp_risk + 0.5, 0, 1)
    return float(risk)

def get_alert_level(risk):
    if risk < 0.4:
        return "LOW"
    elif risk < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"

# ---------------- UI ----------------
st.set_page_config(page_title="ICU Deterioration Simulator", layout="wide")

st.sidebar.title("ICU Patient Deterioration Simulator")
selected_patient = st.sidebar.selectbox("Select Patient", patient_ids)
selected_hour = st.sidebar.slider("Hour", WINDOW_SIZE, MAX_HOUR)

st.title(f"Patient {selected_patient} Overview")

# ---------------- DATA SIMULATION ----------------
vitals_df = generate_fake_vitals(selected_hour)
risk_score = simulate_risk_score(vitals_df)
alert = get_alert_level(risk_score)

# ---------------- DISPLAY ----------------
st.subheader("Patient Vital Trends")
st.line_chart(vitals_df)

st.subheader("Deterioration Risk")
st.metric("Risk Score", f"{risk_score:.2f}")

# ---------------- ALERT ----------------
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
