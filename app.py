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

btn_stat = st.sidebar.button(
    "📊 Statistika Deskriptif"
)

btn_plot = st.sidebar.button(
    "📈 Visualisasi Time Series Plot"
)

btn_missing = st.sidebar.button(
    "🧩 Cek Missing Values"
)

btn_outlier = st.sidebar.button(
    "🚨 Cek Outliers"
)

btn_baseline = st.sidebar.button(
    "🤖 Baseline Model (GRU-Adam)"
)

btn_forecast = st.sidebar.button(
    "🔮 Forecast"
)

btn_compare = st.sidebar.button(
    "📉 Grafik Predict vs Actual"
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

    try:

        # ==============================================
        # MEMBACA FILE EXCEL
        # ==============================================
        df = pd.read_excel(uploaded_file)

        # Hapus spasi pada nama kolom
        df.columns = df.columns.str.strip()

        st.success("✅ File Excel berhasil dibaca!")

        # ==============================================
        # INFORMASI DATASET
        # ==============================================
        st.subheader("📌 Informasi Dataset")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Jumlah Baris",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Jumlah Kolom",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                df.isnull().sum().sum()
            )

        # ==============================================
        # PREVIEW DATA
        # ==============================================
        st.subheader("📄 Preview Dataset")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # ==============================================
        # INFORMASI TIPE DATA
        # ==============================================
        st.subheader("📌 Tipe Data")

        dtype_df = pd.DataFrame({
            "Kolom": df.columns,
            "Tipe Data": df.dtypes.astype(str)
        })

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

        # ==============================================
        # VALIDASI KOLOM
        # ==============================================
        required_cols = [
            "Tanggal",
            "Terakhir"
        ]

        missing_cols = [
            col for col in required_cols
            if col not in df.columns
        ]

        if len(missing_cols) > 0:

            st.error(
                f"""
                Kolom berikut tidak ditemukan:
                {missing_cols}
                """
            )

        else:

            st.success(
                "✅ Kolom 'Tanggal' dan 'Terakhir' tersedia"
            )

    except Exception as e:

        st.error(
            f"❌ Terjadi error saat membaca file: {e}"
        )

else:

    st.info(
        "📂 Silakan upload file Excel terlebih dahulu."
    )

# ======================================================
# ANALISIS DATA
# ======================================================
if btn_stat:

    st.subheader("📊 Statistika Deskriptif")

    st.write(df.describe())

# ======================================================
# TIME SERIES PLOT
# ======================================================
if btn_plot:

    st.subheader("📈 Visualisasi Time Series Plot")

    if "Tanggal" in df.columns and "Terakhir" in df.columns:

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            pd.to_datetime(df["Tanggal"]),
            df["Terakhir"]
        )

        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga")
        ax.set_title("Time Series Harga Emas")

        st.pyplot(fig)

    else:

        st.warning(
            "Kolom 'Tanggal' dan 'Terakhir' tidak ditemukan."
        )

# ======================================================
# MISSING VALUE
# ======================================================
if btn_missing:

    st.subheader("🧩 Cek Missing Values")

    missing_df = pd.DataFrame({
        "Kolom": df.columns,
        "Jumlah Missing": df.isnull().sum().values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

# ======================================================
# OUTLIER
# ======================================================
if btn_outlier:

    st.subheader("🚨 Deteksi Outliers (Metode IQR)")

    if "Terakhir" in df.columns:

        Q1 = df["Terakhir"].quantile(0.25)
        Q3 = df["Terakhir"].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        outliers = df[
            (df["Terakhir"] < lower_bound) |
            (df["Terakhir"] > upper_bound)
        ]

        st.write(f"Jumlah Outlier: {len(outliers)}")

        st.dataframe(
            outliers,
            use_container_width=True
        )

    else:

        st.warning(
            "Kolom 'Terakhir' tidak ditemukan."
        )

# ======================================================
# BASELINE MODEL
# ======================================================
if btn_baseline:

    st.subheader("🤖 Baseline Model (GRU-Adam)")

    st.info(
        """
        Baseline model menggunakan:
        
        - Optimizer : Adam
        - Loss Function : Mean Squared Error (MSE)
        - Activation : tanh
        - Dense Output : linear
        """
    )

# ======================================================
# FORECAST
# ======================================================
if btn_forecast:

    st.subheader("🔮 Forecast")

    st.info(
        """
        Forecast digunakan untuk memprediksi
        harga emas pada periode mendatang
        berdasarkan pola historis data.
        """
    )

# ======================================================
# PREDICT VS ACTUAL
# ======================================================
if btn_compare:

    st.subheader("📉 Grafik Predict vs Actual")

    st.info(
        """
        Grafik ini digunakan untuk membandingkan:
        
        - Data aktual
        - Data hasil prediksi model
        """
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
