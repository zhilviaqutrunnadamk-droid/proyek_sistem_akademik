import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Pengaturan halaman (harus selalu di paling atas)
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓"
)

# Fungsi melatih model dengan cache agar tidak dilatih ulang setiap klik
@st.cache_resource
def train_model():
    np.random.seed(42)
    
    # Rentang diubah menjadi 0-100 agar sesuai dengan input pengguna
    data = pd.DataFrame({
        "Kehadiran": np.random.randint(0, 101, 1000),
        "Tugas": np.random.randint(0, 101, 1000),
        "UTS": np.random.randint(0, 101, 1000),
        "UAS": np.random.randint(0, 101, 1000)
    })

    nilai = (
        data["Kehadiran"] * 0.10 +
        data["Tugas"] * 0.20 +
        data["UTS"] * 0.30 +
        data["UAS"] * 0.40
    )

    data["Status"] = np.where(nilai >= 60, 1, 0)

    X = data[["Kehadiran", "Tugas", "UTS", "UAS"]]
    y = data["Status"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)
    return model

# Memanggil model yang sudah di-cache
model = train_model()

# Tampilan website
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk mengetahui hasil prediksi kelulusan.")

st.divider()

# Menggunakan st.form agar halaman tidak reload setiap kali mengubah satu angka
with st.form("form_prediksi"):
    nama = st.text_input("Nama Mahasiswa")
    
    col1, col2 = st.columns(2)
    with col1:
        kehadiran = st.number_input("Nilai Kehadiran", 0, 100, 80)
        tugas = st.number_input("Nilai Tugas", 0, 100, 80)
    with col2:
        uts = st.number_input("Nilai UTS", 0, 100, 80)
        uas = st.number_input("Nilai UAS", 0, 100, 80)
        
    submit = st.form_submit_button("Prediksi Kelulusan")

# Logika setelah tombol ditekan
if submit:
    if nama.strip() == "":
        st.warning("Silakan isi nama mahasiswa terlebih dahulu.")
    else:
        input_data = pd.DataFrame(
            [[kehadiran, tugas, uts, uas]],
            columns=["Kehadiran", "Tugas", "UTS", "UAS"]
        )

        hasil = model.predict(input_data)[0]
        probabilitas = model.predict_proba(input_data)[0][hasil] * 100

        st.divider()
        st.subheader("Hasil Prediksi")
        st.write(f"**Nama Mahasiswa:** {nama}")

        if hasil == 1:
            st.success("SELAMAT! DIPREDIKSI LULUS 🎉")
        else:
            st.error("DIPREDIKSI BELUM LULUS")

        st.write(f"**Tingkat keyakinan model:** {round(probabilitas, 2)}%")
        st.caption("Hasil merupakan simulasi model AI, bukan keputusan kelulusan resmi.")