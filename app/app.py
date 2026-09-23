import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="centered"
)


# Load trained model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_balanced_pipeline.pkl"

model = joblib.load(MODEL_PATH)


# Title
st.title("⚙️ Predictive Maintenance & Equipment Failure Prediction")
st.write(
    "Enter the machine operating conditions below to predict "
    "the probability of equipment failure."
)
# Machine inputs
st.subheader("Machine Parameters")

machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.1
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=0,
    max_value=5000,
    value=1500,
    step=10
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.1
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)
# Prediction button
if st.button("Predict Failure Risk"):

    # Feature engineering
    temperature_difference = (
        process_temperature - air_temperature
    )

    mechanical_power = (
        torque * rotational_speed
    )

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Temperature_Difference": [temperature_difference],
        "Mechanical_Power": [mechanical_power]
    })

    # Predict failure probability
    failure_probability = model.predict_proba(input_data)[0][1]

    # Predict failure class
    prediction = model.predict(input_data)[0]

    # Display prediction probability
    st.subheader("Prediction Result")

    st.metric(
        "Failure Probability",
        f"{failure_probability * 100:.2f}%"
    )
        # Calculate risk level
    if failure_probability < 0.30:
        risk_level = "Low Risk"
    elif failure_probability < 0.70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    # Display risk level
    st.subheader("Risk Assessment")
    st.write(f"**Risk Level:** {risk_level}")
        # Display feature importance
    st.subheader("Model Feature Importance")

    importance = model.named_steps["model"].feature_importances_
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(
        feature_importance_df,
        use_container_width=True
    )