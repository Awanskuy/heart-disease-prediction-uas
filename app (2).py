import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size:2rem; font-weight:700; color:#1F4E79; text-align:center; margin-bottom:0.2rem; }
    .sub-title { font-size:0.95rem; color:#666; text-align:center; margin-bottom:1.5rem; }
    .result-pos { background:#FCE4D6; border-left:5px solid #E15759; padding:1rem; border-radius:8px; margin:1rem 0; }
    .result-neg { background:#E2EFDA; border-left:5px solid #70AD47; padding:1rem; border-radius:8px; margin:1rem 0; }
    .info-box { background:#EBF3FB; border-left:4px solid #2E75B6; padding:0.8rem; border-radius:6px; font-size:0.85rem; color:#444; margin:0.8rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {}
    paths = {
        "Random Forest": "model/random_forest.pkl",
        "XGBoost":       "model/xgboost.pkl",
    }
    for name, path in paths.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models

models = load_models()

st.markdown('<div class="main-title">❤️ Prediksi Penyakit Jantung</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Muhammad Nasywan Amin · NIM 241730084 · UAS Kecerdasan Buatan</div>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Model")
    if models:
        model_choice = st.selectbox("Pilih Model", list(models.keys()))
    else:
        st.error("Model tidak ditemukan.")
        model_choice = None

    st.divider()
    st.markdown("### Performa Model")
    st.markdown("""
| Model | Accuracy | AUC |
|---|---|---|
| Random Forest | 0.8660 | 0.9260 |
| XGBoost | 0.8606 | 0.9207 |

*Stratified 5-Fold CV · 918 record*
    """)
    st.divider()
    st.markdown('<div class="info-box">Aplikasi ini untuk keperluan akademis. Bukan pengganti diagnosis medis.</div>', unsafe_allow_html=True)

st.markdown("### Data Pasien")
col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Usia (tahun)", 20, 100, 55)
    sex      = st.selectbox("Jenis Kelamin", [1,0], format_func=lambda x: "Laki-laki" if x==1 else "Perempuan")
    cp       = st.selectbox("Tipe Nyeri Dada", [0,1,2,3], format_func=lambda x: {0:"0-Typical angina",1:"1-Atypical angina",2:"2-Non-anginal",3:"3-Asymptomatic"}[x])
    trestbps = st.number_input("Tekanan Darah (mmHg)", 80, 220, 130)
    chol     = st.number_input("Kolesterol (mg/dl)", 100, 600, 240)
    fbs      = st.selectbox("Gula Darah > 120 mg/dl", [0,1], format_func=lambda x: "Ya" if x==1 else "Tidak")

with col2:
    restecg  = st.selectbox("Hasil EKG", [0,1,2], format_func=lambda x: {0:"0-Normal",1:"1-ST-T abnormal",2:"2-LV hypertrophy"}[x])
    thalach  = st.number_input("Detak Jantung Maks", 60, 220, 150)
    exang    = st.selectbox("Angina saat olahraga", [0,1], format_func=lambda x: "Ya" if x==1 else "Tidak")
    oldpeak  = st.number_input("ST depression", 0.0, 8.0, 1.0, 0.1)
    slope    = st.selectbox("Slope ST", [0,1,2], format_func=lambda x: {0:"0-Upsloping",1:"1-Flat",2:"2-Downsloping"}[x])

st.divider()

if st.button("Prediksi Sekarang", type="primary", use_container_width=True):
    if not models or model_choice not in models:
        st.error("Model tidak tersedia.")
    else:
        input_df = pd.DataFrame(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]],
            columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope"]
        )
        model = models[model_choice]
        pred  = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.markdown("### Hasil Prediksi")
        if pred == 1:
            st.markdown(f'<div class="result-pos"><b>Terdeteksi Risiko Penyakit Jantung</b><br>Model {model_choice} mendeteksi adanya indikasi penyakit jantung. Segera konsultasikan dengan dokter.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-neg"><b>Tidak Terdeteksi Penyakit Jantung</b><br>Model {model_choice} tidak mendeteksi indikasi penyakit jantung berdasarkan data yang dimasukkan.</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Model", model_choice)
        c2.metric("Prob. Positif", f"{proba[1]:.1%}")
        c3.metric("Prob. Negatif", f"{proba[0]:.1%}")

        st.bar_chart(pd.DataFrame({
            "Kelas": ["Tidak Ada Penyakit", "Ada Penyakit"],
            "Probabilitas": [proba[0], proba[1]]
        }).set_index("Kelas"))

        with st.expander("Data Input"):
            st.dataframe(pd.DataFrame({
                "Fitur": ["Usia","Jenis Kelamin","Nyeri Dada","Tek. Darah","Kolesterol",
                          "Gula Darah","EKG","Detak Maks","Angina","ST Depression","Slope ST"],
                "Nilai": [age, "Laki-laki" if sex==1 else "Perempuan", cp,
                          trestbps, chol, "Ya" if fbs==1 else "Tidak",
                          restecg, thalach, "Ya" if exang==1 else "Tidak", oldpeak, slope]
            }), use_container_width=True, hide_index=True)

        st.markdown('<div class="info-box">Hasil prediksi ini untuk keperluan akademis dan tidak dapat dijadikan diagnosa medis.</div>', unsafe_allow_html=True)

st.divider()
st.markdown('<div style="text-align:center;color:#999;font-size:0.8rem;">UAS Kecerdasan Buatan · Teknik Informatika · UIN Sultan Maulana Hasanuddin Banten<br>Muhammad Nasywan Amin (241730084)</div>', unsafe_allow_html=True)
