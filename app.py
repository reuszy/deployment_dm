# 5. Buat Dashboard

import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Heart Failure Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model
model_path = 'model_gagal_jantung.pkl'

try:
    model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    st.error("Model tidak ditemukan. Harap jalankan training dulu.")
    st.stop()

# Header
st.title("Clinical Heart Failure Prediction System")
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
Aplikasi ini menggunakan algoritma **Random Forest** untuk memprediksi risiko kematian akibat gagal jantung
berdasarkan 12 fitur klinis pasien. Silakan isi data rekam medis di bawah ini.
---
""")

#Body
st.subheader("Input Data Medis Pasien")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("Data User")
    age = st.number_input('Umur (Tahun)', 40, 95, 60)
    sex = st.selectbox('Jenis Kelamin', (1, 0), format_func=lambda x: 'Pria' if x == 1 else 'Wanita')
    smoking = st.selectbox('Perokok Aktif?', (0, 1), format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
    time = st.number_input('Waktu Follow-up (Hari)', 4, 285, 150, help="Periode waktu pengamatan pasien dalam hari")

with col2:
    st.info("Riwayat Penyakit")
    diabetes = st.selectbox('Riwayat Diabetes', (0, 1), format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
    high_blood_pressure = st.selectbox('Darah Tinggi (Hipertensi)', (0, 1), format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
    anaemia = st.selectbox('Anaemia (Kurang Darah)', (0, 1), format_func=lambda x: 'Ya' if x == 1 else 'Tidak')
    ejection_fraction = st.slider('Ejection Fraction (%)', 14, 80, 38, help="Persentase darah yang meninggalkan jantung setiap kontraksi")

with col3:
    st.info("Hasil Laboratorium")
    creatinine_phosphokinase = st.number_input('CPK Enzyme (mcg/L)', 23, 7861, 582)
    platelets = st.number_input('Platelets (kilo-platelets/mL)', 25000.0, 850000.0, 263358.0)
    serum_creatinine = st.number_input('Serum Creatinine (mg/dL)', 0.5, 9.4, 1.1)
    serum_sodium = st.number_input('Serum Sodium (mEq/L)', 113, 148, 137)

input_data = {
    'age': age, 'anaemia': anaemia, 'creatinine_phosphokinase': creatinine_phosphokinase,
    'diabetes': diabetes, 'ejection_fraction': ejection_fraction, 'high_blood_pressure': high_blood_pressure,
    'platelets': platelets, 'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium,
    'sex': sex, 'smoking': smoking, 'time': time
}
input_df = pd.DataFrame(input_data, index=[0])

# Button
st.markdown("---")
if st.button('ANALISIS RISIKO SEKARANG', use_container_width=True):

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)
    prob_meninggal = probability[0][1] * 100
    prob_selamat = probability[0][0] * 100

    st.subheader("Hasil Analisis Prediksi")

    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction[0] == 1:
            st.error(f"### RISIKO TINGGI (High Risk)")
            st.write(f"Pasien memiliki probabilitas **{prob_meninggal:.2f}%** mengalami kegagalan jantung fatal.")
            st.write("Saran: Segera rujuk ke spesialis kardiovaskular untuk penanganan intensif.")
        else:
            st.success(f"### RISIKO RENDAH (Low Risk)")
            st.write(f"Pasien diprediksi aman dengan tingkat keyakinan **{prob_selamat:.2f}%**.")
            st.write("Saran: Tetap jaga pola hidup sehat dan kontrol rutin.")

    with res_col2:
        st.write("### Tingkat Risiko")
        st.progress(int(prob_meninggal))
        st.caption(f"Risk Meter: {prob_meninggal:.2f}%")

    with st.expander("Lihat Detail Data Input Pasien"):
        st.dataframe(input_df)

# Sidebar
st.sidebar.title("Tentang Aplikasi")
st.sidebar.info("Aplikasi ini dibuat sebagai Tugas Data Mining.")
st.sidebar.write("**Fitur Utama:**")
st.sidebar.markdown("""
- Prediksi Real-time
- Kalkulasi Probabilitas
- Indikator Risiko Visual
""")
st.sidebar.caption("© 2025 Rizqi Dwi Saputra - S1SI-07-C")