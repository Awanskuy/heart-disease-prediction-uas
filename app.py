import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── STYLE ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700; color: #1F4E79;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem; color: #666; text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-box-positive {
        background: #FCE4D6; border-left: 5px solid #E15759;
        padding: 1rem 1.2rem; border-radius: 8px; margin: 1rem 0;
    }
    .result-box-negative {
        background: #E2EFDA; border-left: 5px solid #70AD47;
        padding: 1rem 1.2rem; border-radius: 8px; margin: 1rem 0;
    }
    .result-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.3rem; }
    .metric-card {
        background: #EDF2F9; border-radius: 8px;
        padding: 0.8rem 1rem; text-align: center; margin: 0.3rem 0;
    }
    .info-box {
        background: #EBF3FB; border-left: 4px solid #2E75B6;
        padding: 0.8rem 1rem; border-radius: 6px;
        font-size: 0.85rem; color: #444; margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    paths = {
        "Random Forest":  "model/random_forest.pkl",
        "XGBoost":        "model/xgboost.pkl",
    }
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

models = load_models()

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🫀 Prediksi Penyakit Jantung</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Muhammad Nasywan Amin · NIM 241730084 · UAS Kecerdasan Buatan</div>', unsafe_allow_html=True)
st.divider()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan Model")
    if models:
        model_choice = st.selectbox("Pilih Model", list(models.keys()))
    else:
        st.error("Model tidak ditemukan. Pastikan file .pkl ada di folder model/")
        model_choice = None

    st.divider()
    st.markdown("### 📊 Informasi Model")
    st.markdown("""
    | Model | Accuracy | AUC |
    |---|---|---|
    | Random Forest | 0.8660 | 0.9260 |
    | XGBoost | 0.8606 | 0.9207 |

    *Evaluasi: Stratified 5-Fold CV*
    *Dataset: UCI Heart Disease (918 record)*
    """)

    st.divider()
    st.markdown("""
    <div class="info-box">
    ⚠️ Aplikasi ini hanya untuk keperluan akademis.
    Bukan pengganti diagnosis medis profesional.
    </div>
    """, unsafe_allow_html=True)

# ── INPUT FORM ───────────────────────────────────────────────────────────────
st.markdown("### 📋 Data Pasien")
st.markdown("Isi data klinis pasien di bawah ini:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia (tahun)", min_value=20, max_value=100, value=55, step=1)
    sex = st.selectbox("Jenis Kelamin", options=[1, 0], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    cp = st.selectbox("Tipe Nyeri Dada (cp)",
                      options=[0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "0 - Typical angina",
                          1: "1 - Atypical angina",
                          2: "2 - Non-anginal pain",
                          3: "3 - Asymptomatic"
                      }[x])
    trestbps = st.number_input("Tekanan Darah Istirahat (mmHg)", min_value=80, max_value=220, value=130, step=1)
    chol = st.number_input("Kolesterol Serum (mg/dl)", min_value=100, max_value=600, value=240, step=1)
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl (fbs)",
                       options=[0, 1],
                       format_func=lambda x: "Ya (> 120 mg/dl)" if x == 1 else "Tidak (≤ 120 mg/dl)")

with col2:
    restecg = st.selectbox("Hasil EKG Istirahat (restecg)",
                           options=[0, 1, 2],
                           format_func=lambda x: {
                               0: "0 - Normal",
                               1: "1 - ST-T abnormality",
                               2: "2 - LV hypertrophy"
                           }[x])
    thalach = st.number_input("Detak Jantung Maksimum", min_value=60, max_value=220, value=150, step=1)
    exang = st.selectbox("Angina akibat olahraga (exang)",
                         options=[0, 1],
                         format_func=lambda x: "Ya" if x == 1 else "Tidak")
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=8.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope ST segment",
                         options=[0, 1, 2],
                         format_func=lambda x: {
                             0: "0 - Upsloping",
                             1: "1 - Flat",
                             2: "2 - Downsloping"
                         }[x])

# ── PREDIKSI ─────────────────────────────────────────────────────────────────
st.divider()

if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
    if not models or model_choice not in models:
        st.error("Model tidak tersedia. Upload file .pkl ke folder model/")
    else:
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                    restecg, thalach, exang, oldpeak, slope]],
                                  columns=["age","sex","cp","trestbps","chol","fbs",
                                           "restecg","thalach","exang","oldpeak","slope"])

        model = models[model_choice]
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        prob_positive = proba[1]
        prob_negative = proba[0]

        st.markdown("### 🎯 Hasil Prediksi")

        if pred == 1:
            st.markdown(f"""
            <div class="result-box-positive">
                <div class="result-title">❗ Terdeteksi Risiko Penyakit Jantung</div>
                <p>Model <b>{model_choice}</b> mendeteksi adanya indikasi penyakit jantung.
                Disarankan untuk berkonsultasi dengan tenaga medis profesional.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-negative">
                <div class="result-title">✅ Tidak Terdeteksi Penyakit Jantung</div>
                <p>Model <b>{model_choice}</b> tidak mendeteksi indikasi penyakit jantung
                berdasarkan data yang dimasukkan.</p>
            </div>
            """, unsafe_allow_html=True)

        # Probabilitas
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Model", model_choice)
        with col_b:
            st.metric("Prob. Positif", f"{prob_positive:.1%}")
        with col_c:
            st.metric("Prob. Negatif", f"{prob_negative:.1%}")

        # Bar probabilitas
        st.markdown("**Probabilitas Prediksi:**")
        prob_df = pd.DataFrame({
            "Kelas": ["Tidak Ada Penyakit (0)", "Ada Penyakit (1)"],
            "Probabilitas": [prob_negative, prob_positive]
        })
        st.bar_chart(prob_df.set_index("Kelas"))

        # Ringkasan input
        with st.expander("📄 Ringkasan Data Input"):
            input_display = pd.DataFrame({
                "Fitur": ["Usia", "Jenis Kelamin", "Tipe Nyeri Dada", "Tek. Darah",
                          "Kolesterol", "Gula Darah", "EKG", "Detak Maks",
                          "Angina Olahraga", "ST Depression", "Slope ST"],
                "Nilai": [age,
                          "Laki-laki" if sex == 1 else "Perempuan",
                          cp, trestbps, chol,
                          "Ya" if fbs == 1 else "Tidak",
                          restecg, thalach,
                          "Ya" if exang == 1 else "Tidak",
                          oldpeak, slope]
            })
            st.dataframe(input_display, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info-box">
        ⚠️ <b>Disclaimer:</b> Hasil prediksi ini dihasilkan oleh model machine learning
        untuk keperluan akademis dan tidak dapat dijadikan diagnosa medis.
        Selalu konsultasikan kondisi kesehatan Anda dengan dokter.
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem;">
UAS Kecerdasan Buatan · Program Studi Teknik Informatika · UIN Sultan Maulana Hasanuddin Banten<br>
Muhammad Nasywan Amin (241730084) · Dataset: UCI Heart Disease Gabungan (918 record)
</div>
""", unsafe_allow_html=True)
