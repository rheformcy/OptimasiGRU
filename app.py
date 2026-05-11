import streamlit as st
import pandas as pd

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Optimasi GRU",
    layout="wide"
)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
# 📊 Optimasi Gated Recurrent Unit (GRU)

### Rhena Amelia Shafitry  
Statistika UNDIP  
24050122120019
""")

st.divider()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload File Excel",
    type=["xlsx", "xls"]
)

st.sidebar.caption(
    "Format file harus Excel (.xlsx / .xls)"
)

# ======================================================
# PILIHAN ANALISIS
# ======================================================
st.sidebar.header("📌 Pilihan Analisis")

check_stat = st.sidebar.checkbox(
    "Statistika Deskriptif"
)

check_plot = st.sidebar.checkbox(
    "Visualisasi Time Series Plot"
)

check_missing = st.sidebar.checkbox(
    "Cek Missing Values"
)

check_outlier = st.sidebar.checkbox(
    "Cek Outliers"
)

check_baseline = st.sidebar.checkbox(
    "Baseline Model (GRU-Adam)"
)

check_forecast = st.sidebar.checkbox(
    "Forecast"
)

check_compare = st.sidebar.checkbox(
    "Grafik Perbandingan Predict vs Actual"
)

# ======================================================
# PARAMETER GRU-PSO
# ======================================================
st.sidebar.header("⚙️ Optimasi GRU-PSO")

iterasi = st.sidebar.number_input(
    "Jumlah Iterasi",
    min_value=1,
    value=10
)

particle = st.sidebar.number_input(
    "Jumlah Particle",
    min_value=1,
    value=20
)

epoch = st.sidebar.number_input(
    "Jumlah Epoch",
    min_value=1,
    value=50
)

learning_rate = st.sidebar.number_input(
    "Learning Rate",
    min_value=0.0001,
    max_value=1.0,
    value=0.0010,
    step=0.0001,
    format="%.4f"
)

batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=1,
    value=32
)

dropout = st.sidebar.slider(
    "Dropout",
    min_value=0.0,
    max_value=0.9,
    value=0.2,
    step=0.1
)

layer = st.sidebar.number_input(
    "Jumlah Layer",
    min_value=1,
    max_value=5,
    value=1
)

units = st.sidebar.number_input(
    "Jumlah Units",
    min_value=1,
    value=64
)

timestep = st.sidebar.number_input(
    "Timestep / Window Size",
    min_value=1,
    value=3
)

# ======================================================
# BUTTON
# ======================================================
run_btn = st.sidebar.button(
    "🚀 Jalankan Model"
)

# ======================================================
# HALAMAN UTAMA
# ======================================================
if uploaded_file is not None:

    st.success("File berhasil diupload!")

    try:
        df = pd.read_excel(uploaded_file)

        st.subheader("📄 Preview Dataset")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Terjadi error: {e}")

else:

    st.info(
        "Silakan upload dataset Excel terlebih dahulu."
    )

# ======================================================
# INFORMASI PARAMETER
# ======================================================
if run_btn:

    st.divider()

    st.subheader("📌 Konfigurasi Model")

    col1, col2 = st.columns(2)

    with col1:

        st.write(f"**Iterasi PSO:** {iterasi}")
        st.write(f"**Particle:** {particle}")
        st.write(f"**Epoch:** {epoch}")
        st.write(f"**Learning Rate:** {learning_rate}")
        st.write(f"**Batch Size:** {batch_size}")

    with col2:

        st.write(f"**Dropout:** {dropout}")
        st.write(f"**Layer:** {layer}")
        st.write(f"**Units:** {units}")
        st.write(f"**Timestep:** {timestep}")

    st.success(
        "UI berhasil dijalankan. "
        "Tahap selanjutnya tinggal integrasi model GRU-PSO."
    )
