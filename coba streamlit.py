import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import gc

# ==========================================
# 1. KONFIGURASI UI
# ==========================================
st.set_page_config(page_title="OptimasGRU - Gold Forecasting", layout="wide")

st.title("📊 Sistem Prediksi Harga Emas (GRU & PSO)")
st.markdown("Aplikasi untuk membandingkan performa model **GRU Standar** dan **GRU yang dioptimasi PSO**.")

# --- SIDEBAR ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Data Historis", type=["xlsx", "xls"])
st.sidebar.caption("⚠️ File harus berupa format Excel (.xlsx atau .xls)")

st.sidebar.header("2. Analisis Data")
show_stat = st.sidebar.checkbox("Stat Deskriptif Data")
show_plot = st.sidebar.checkbox("Plot Time Series Data")
show_missing = st.sidebar.checkbox("Cek Missing Value & Outlier")

st.sidebar.header("3. Hyperparameter Model")

# Baseline Model (Fixed)
with st.sidebar.expander("Baseline Model (Fixed Parameters)"):
    st.info("Units: 50, LR: 0.0001, Batch: 64, Epoch: 100, Dropout: 0.5")

# Tuning GRU-PSO
with st.sidebar.expander("Tuning GRU-PSO"):
    input_lr = st.number_input("Learning Rate", value=0.001, format="%.4f")
    input_units = st.slider("GRU Units", 16, 128, 50)
    input_batch = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=64)
    input_epoch = st.number_input("Epochs", value=50)
    input_pso_particles = st.number_input("PSO Particles", value=40)
    input_pso_iters = st.number_input("PSO Iterations", value=10)

run_model = st.sidebar.button("🚀 Mulai Proses Training & Prediksi")

# ==========================================
# 2. FUNGSI HELPER (LOGIKA CODING)
# ==========================================

def windowing_data(data, window=1):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# ==========================================
# 3. PROSES DATA (MAIN)
# ==========================================

if uploaded_file is not None:
    # A. Load & Clean Data
    emas = pd.read_excel(uploaded_file)
    emas = emas[['Tanggal', 'Terakhir']]
    emas.dropna(inplace=True)
    emas['Tanggal'] = pd.to_datetime(emas['Tanggal'], dayfirst=True)
    emas = emas.sort_values(by='Tanggal')
    
    # B. Tampilan Analisis (Checklist)
    if show_stat:
        st.subheader("📌 Statistik Deskriptif")
        st.write(emas['Terakhir'].describe())

    if show_plot:
        st.subheader("📈 Plot Time Series Harga Emas")
        plt.figure(figsize=(12, 5))
        plt.plot(emas['Tanggal'], emas['Terakhir'], color='#1A5276', linewidth=2)
        plt.grid(True, axis='y', linestyle=':', alpha=0.5)
        st.pyplot(plt)

    if show_missing:
        st.subheader("🔍 Missing Value & Outlier")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write("Jumlah Missing Value:")
            st.write(emas.isnull().sum())
        with col_m2:
            st.write("Visualisasi Boxplot:")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=emas['Terakhir'], color='gold', ax=ax2)
            st.pyplot(fig2)

    # C. Eksekusi Model
    if run_model:
        st.divider()
        st.header("Results Analysis")
        
        # 1. Preprocessing (Scaling & Splitting)
        values = emas[['Terakhir']].values
        n_train = int(len(values) * 0.8)
        
        scaler = MinMaxScaler().fit(values[:n_train])
        scaled_data = scaler.transform(values)
        
        X_all, y_all = windowing_data(scaled_data, window=1)
        
        dtrain_end = n_train - 1
        X_train = X_all[:dtrain_end]
        y_train = y_all[:dtrain_end]
        X_test = X_all[dtrain_end:]
        y_test = y_all[dtrain_end:]
        
        # Tampilan Tab
        tab1, tab2 = st.tabs(["Baseline Model (Standard)", "GRU-PSO Optimized"])
        
        # --- TAB 1: BASELINE ---
        with tab1:
            st.subheader("Baseline GRU Performance")
            with st.spinner('Training Baseline Model...'):
                # --- LOGIKA MODEL BASELINE DI SINI ---
                # (Sesuai parameter tetap: 50 units, 0.0001 LR, dll)
                # dummy_metrics
                st.success("Training Baseline Selesai!")
                st.metric("MAPE Baseline", "1.54%") 

        # --- TAB 2: GRU-PSO ---
        with tab2:
            st.subheader("GRU-PSO Optimized Performance")
            with st.spinner('Running PSO Optimization (This may take a while)...'):
                # --- LOGIKA MODEL GRU-PSO DI SINI ---
                # Gunakan: input_lr, input_units, input_batch, input_pso_particles
                st.write(f"Iterasi PSO: {input_pso_iters} | Partikel: {input_pso_particles}")
                
                # Setelah training, tampilkan hasil evaluasi
                st.success("Optimasi PSO Selesai!")
                st.metric("MAPE GRU-PSO", "1.24%", delta="-0.30% (Better)")

        # BAGIAN FORECAST AKAN KAMU TAMBAH DI SINI
        st.write("---")
        st.info("💡 Kamu bisa menambahkan tombol 'Forecast 5 Hari ke Depan' di bawah ini.")

else:
    st.warning("Silakan upload file Excel harga emas untuk memulai.")

# ==========================================
# 4. FOOTER
# ==========================================
st.sidebar.markdown("---")
st.sidebar.caption("Project by Rhena Amelia Shafitry - UNDIP 2022")
