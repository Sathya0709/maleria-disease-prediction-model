import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Malaria Prediction",
    page_icon="🧪",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load("/content/malaria_random_forest_model.pkl")
    features = joblib.load("/content/malaria_feature_names.pkl")
    return model, features

model, features = load_model()

st.title("🧪 Malaria Disease Prediction")
st.write(
    "Educational machine-learning demonstration "
    "using a synthetic dataset."
)

st.warning(
    "This application is not a medical diagnostic tool. "
    "Its labels are synthetic and must not guide medical decisions."
)

st.subheader("Enter Sample Details")

with st.form("prediction_form"):
    age = st.number_input(
        "Age (years)", min_value=1, max_value=100, value=25
    )

    col1, col2 = st.columns(2)

    with col1:
        fever = st.selectbox("Fever", [0, 1], format_func=lambda x: "Yes" if x else "No")
        chills = st.selectbox("Chills", [0, 1], format_func=lambda x: "Yes" if x else "No")
        headache = st.selectbox("Headache", [0, 1], format_func=lambda x: "Yes" if x else "No")
        sweating = st.selectbox("Sweating", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col2:
        fatigue = st.selectbox("Fatigue", [0, 1], format_func=lambda x: "Yes" if x else "No")
        nausea = st.selectbox("Nausea", [0, 1], format_func=lambda x: "Yes" if x else "No")
        hemoglobin = st.number_input(
            "Hemoglobin (g/dL)", min_value=3.0, max_value=25.0,
            value=11.0, step=0.1
        )
        platelets = st.number_input(
            "Platelet count (/µL)", min_value=1000,
            max_value=1000000, value=140000, step=1000
        )

    submitted = st.form_submit_button("Predict Dataset Class")

if submitted:
    sample = {
        "Age": age,
        "Fever": fever,
        "Chills": chills,
        "Headache": headache,
        "Sweating": sweating,
        "Fatigue": fatigue,
        "Nausea": nausea,
        "Hemoglobin_g_dL": hemoglobin,
        "Platelet_Count_per_uL": platelets
    }

    input_df = pd.DataFrame([sample])[features]
    prediction = int(model.predict(input_df)[0])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Predicted dataset class: 1 (synthetic positive label)")
    else:
        st.success("Predicted dataset class: 0 (synthetic negative label)")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        st.write("Model probability estimates for the synthetic labels:")
        for label, probability in zip(model.classes_, probabilities):
            st.write(f"Class {label}: {probability:.1%}")

    st.caption(
        "These results are model outputs on synthetic labels, "
        "not estimates of actual malaria status."
    )
