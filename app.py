"""
Road Accident Risk Predictor — Streamlit interface
Run locally with:  streamlit run app.py

Expects `accident_risk_model.pkl` (produced by the Colab notebook, step 14)
to sit in the same folder as this file.
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Road Accident Risk Predictor", page_icon="🚦", layout="centered")

MODEL_PATH = "accident_risk_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

bundle = load_model()

st.title("🚦 Road Accident Risk Predictor")
st.caption("Enter road conditions below to predict accident risk.")

if bundle is None:
    st.error(
        f"Couldn't find `{MODEL_PATH}` in this folder. "
        "Train the model in the Colab notebook (step 14), download "
        "`accident_risk_model.pkl`, and place it next to `app.py`."
    )
    st.stop()

model = bundle["model"]
feature_cols = bundle["feature_cols"]
categorical_cols = bundle["categorical_cols"]
boolean_cols = bundle["boolean_cols"]
numeric_cols = bundle["numeric_cols"]
threshold = bundle["threshold"]
model_name = bundle["model_name"]
val_accuracy = bundle["val_accuracy"]
val_f1 = bundle["val_f1"]

# --- Sidebar: model performance ---
with st.sidebar:
    st.header("Model info")
    st.write(f"**Model:** {model_name}")
    st.metric("Validation Accuracy", f"{val_accuracy:.2%}")
    st.metric("Validation F1 Score", f"{val_f1:.2%}")
    st.caption(f"Classification threshold on accident_risk: {threshold}")

# --- Input form ---
st.subheader("Road / trip details")

col1, col2 = st.columns(2)

with col1:
    road_type = st.selectbox("Road type", ["urban", "rural", "highway"])
    lighting = st.selectbox("Lighting", ["daylight", "dim", "night"])
    weather = st.selectbox("Weather", ["clear", "rainy", "foggy"])
    time_of_day = st.selectbox("Time of day", ["morning", "afternoon", "evening"])

with col2:
    num_lanes = st.slider("Number of lanes", 1, 6, 2)
    speed_limit = st.slider("Speed limit (mph/kmh as in your data)", 15, 100, 35, step=5)
    curvature = st.slider("Road curvature (0 = straight, 1 = very curved)", 0.0, 1.0, 0.2, step=0.01)
    num_reported_accidents = st.number_input("Previously reported accidents at this spot", 0, 50, 0)

st.subheader("Other conditions")
c3, c4 = st.columns(2)
with c3:
    road_signs_present = st.checkbox("Road signs present", value=True)
    public_road = st.checkbox("Public road", value=True)
with c4:
    holiday = st.checkbox("Holiday", value=False)
    school_season = st.checkbox("School season", value=True)

if st.button("Predict accident risk", type="primary"):
    raw_input = {
        "num_lanes": num_lanes,
        "curvature": curvature,
        "speed_limit": speed_limit,
        "num_reported_accidents": num_reported_accidents,
        "road_signs_present": int(road_signs_present),
        "public_road": int(public_road),
        "holiday": int(holiday),
        "school_season": int(school_season),
        "road_type": road_type,
        "lighting": lighting,
        "weather": weather,
        "time_of_day": time_of_day,
    }

    input_df = pd.DataFrame([raw_input])
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False)

    # align columns exactly with what the model was trained on
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **High Risk** — predicted accident risk probability: {probability:.1%}")
    else:
        st.success(f"✅ **Low Risk** — predicted accident risk probability: {probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        f"Prediction from **{model_name}** "
        f"(validation accuracy {val_accuracy:.1%}, F1 {val_f1:.1%})"
    )