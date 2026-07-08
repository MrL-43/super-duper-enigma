import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediksi Gross Tertiary Education Enrollment",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Prediksi Gross Tertiary Education Enrollment")
st.write(
    """
Aplikasi ini menggunakan model Machine Learning Random Forest
untuk memprediksi **Gross Tertiary Education Enrollment**
berdasarkan indikator pendidikan suatu negara.
"""
)

model = joblib.load("education_model.pkl")

feature_names = model.named_steps["preprocessor"].transformers_[0][2]

st.header("Masukkan Nilai Indikator")

input_data = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        with col1:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.2f"
            )
    else:
        with col2:
            input_data[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.2f"
            )

if st.button("Prediksi"):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    st.success(
        f"Prediksi Gross Tertiary Education Enrollment : {prediction[0]:.2f}%"
    )
