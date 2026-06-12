# 🫀 Prediksi Penyakit Jantung

Aplikasi web prediksi penyakit jantung berbasis Machine Learning menggunakan Random Forest dan XGBoost.

**Muhammad Nasywan Amin · NIM 241730084 · UAS Kecerdasan Buatan**
UIN Sultan Maulana Hasanuddin Banten · Dosen: Ahmad Tabrani, M.T.I.

---

## Fitur Aplikasi

- Input 11 fitur klinis pasien secara interaktif
- Prediksi menggunakan Random Forest atau XGBoost
- Tampilan probabilitas prediksi
- Disclaimer medis

## Dataset

- Sumber: UCI Heart Disease Repository (Kaggle)
- Gabungan: Cleveland, Hungary, Switzerland, Long Beach, Statlog
- Record: 918 (setelah deduplikasi dari 1.190)

## Performa Model

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | 0.8660 | 0.9260 |
| XGBoost | 0.8606 | 0.9207 |

*Evaluasi: Stratified 5-Fold Cross-Validation*

---

## Cara Deploy ke Streamlit Cloud

### 1. Persiapan GitHub

Upload file-file berikut ke repository GitHub:
```
├── app.py
├── requirements.txt
├── README.md
└── model/
    ├── random_forest.pkl
    └── xgboost.pkl
```

### 2. Deploy ke Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun GitHub
3. Klik **New app**
4. Pilih repository dan branch
5. Set **Main file path**: `app.py`
6. Klik **Deploy**

### 3. Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Struktur File

```
app.py              ← Aplikasi Streamlit utama
requirements.txt    ← Dependencies
README.md           ← Dokumentasi ini
model/
  random_forest.pkl ← Model Random Forest terlatih
  xgboost.pkl       ← Model XGBoost terlatih
```

---

## Teknologi

- Python 3.x · Streamlit · scikit-learn 1.6.1 · XGBoost 3.2.0
- Optuna 4.9.0 · Pandas 2.2.2 · NumPy 2.0.2

⚠️ **Disclaimer:** Aplikasi ini untuk keperluan akademis. Bukan pengganti diagnosis medis profesional.
