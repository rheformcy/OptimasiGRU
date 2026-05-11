import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import gc

# ==========================================
# 1. KONFIGURASI UI
# ==========================================
st.set_page_config(page_title="OptimasGRU - Gold Forecasting", layout="wide")

st.title("📊 Sistem Prediksi Harga Emas (GRU & PSO)")
st.markdown("Aplikasi peramalan harga emas menggunakan optimasi **Particle Swarm Optimization**.")

# --- SIDEBAR ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Data Historis", type=["xlsx", "xls"])
st.sidebar.caption("Gunakan file Excel dengan kolom 'Tanggal' dan 'Terakhir'")

st.sidebar.header("2. Analisis Data")
show_stat = st.sidebar.checkbox("Stat Deskriptif Data")
show_plot = st.sidebar.checkbox("Plot Time Series Data")
show_missing = st.sidebar.checkbox("Cek Missing Value & Outlier")

st.sidebar.header("3. Konfigurasi Model")
input_window = st.sidebar.number_input("Window Size (Timestep)", min_value=1, max_value=30, value=1)

with st.sidebar.expander("Tuning GRU-PSO"):
    input_pso_particles = st.number_input("PSO Particles", value=10, min_value=5)
    input_pso_iters = st.number_input("PSO Iterations", value=5, min_value=2)
    input_epochs = st.number_input("Training Epochs", value=50)

run_model = st.sidebar.button("🚀 Mulai Proses Training & Prediksi")

# ==========================================
# 2. FUNGSI LOGIKA (HELPERS)
# ==========================================

def build_model(units, lr, dropout, window):
    model = Sequential([
        Input(shape=(window, 1)),
        GRU(units=int(units), activation='tanh'),
        Dropout(dropout),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def f_fitness(particles, X_tr, y_tr, X_va, y_va, scaler_y, window):
    costs = []
    for p in particles:
        units, lr, batch, dropout = int(p[0]), p[1], int(p[2]), p[3]
        clear_session()
        model = build_model(units, lr, dropout, window)
        # Training singkat untuk evaluasi PSO
        model.fit(X_tr, y_tr, epochs=5, batch_size=int(batch), verbose=0)
        
        y_pred = model.predict(X_va, verbose=0)
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        y_true_inv = scaler_y.inverse_transform(y_va.reshape(-1, 1))
        costs.append(mean_squared_error(y_true_inv, y_pred_inv))
        
        del model
        gc.collect()
    return np.array(costs)

def forecast_future(model, last_input, steps):
    predictions_scaled = []
    current_input = last_input.copy()
    for _ in range(steps):
        pred = model.predict(current_input, verbose=0)
        predictions_scaled.append(pred[0, 0])
        # Update input secara recursive
        new_pred = pred.reshape(1, 1, 1)
        if current_input.shape[1] > 1:
            current_input = np.append(current_input[:, 1:, :], new_pred, axis=1)
        else:
            current_input = new_pred
    return np.array(predictions_scaled).reshape(-1, 1)

# ==========================================
# 3. PROSES UTAMA
# ==========================================
if uploaded_file is not None:
    # Load Data
    emas = pd.read_excel(uploaded_file)
    emas = emas[['Tanggal', 'Terakhir']].dropna()
    emas['Tanggal'] = pd.to_datetime(emas['Tanggal'], dayfirst=True)
    emas = emas.sort_values(by='Tanggal')

    # Preview Analisis
    if show_stat:
        st.subheader("📌 Statistik Deskriptif")
        st.write(emas['Terakhir'].describe())

    if show_plot:
        st.subheader("📈 Tren Harga Historis")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(emas['Tanggal'], emas['Terakhir'], color='#1A5276', label='Harga Emas')
        plt.legend()
        st.pyplot(fig)

    if run_model:
        st.divider()
        
        # 1. Preprocessing
        values = emas[['Terakhir']].values
        n_train = int(len(values) * 0.8)
        scaler = MinMaxScaler().fit(values[:n_train])
        scaled_data = scaler.transform(values)
        
        # Windowing
        X, y = [], []
        for i in range(input_window, len(scaled_data)):
            X.append(scaled_data[i-input_window:i])
            y.append(scaled_data[i])
        X, y = np.array(X), np.array(y)
        
        split_idx = n_train - input_window
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        # 2. Training Tab
        tab1, tab2, tab3 = st.tabs(["Baseline Model", "GRU-PSO Optimized", "Future Forecast"])

        with tab1:
            with st.spinner('Training Baseline...'):
                model_b = build_model(50, 0.0001, 0.5, input_window)
                model_b.fit(X_train, y_train, epochs=int(input_epochs), batch_size=64, verbose=0)
                y_p = model_b.predict(X_test)
                mape_b = mean_absolute_percentage_error(scaler.inverse_transform(y_test), scaler.inverse_transform(y_p)) * 100
                st.metric("MAPE Baseline", f"{mape_b:.4f}%")

        with tab2:
            with st.spinner('Optimasi PSO sedang berjalan...'):
                # PSO Setup
                bounds = (np.array([16, 0.0001, 16, 0.1]), np.array([128, 0.01, 128, 0.5]))
                optimizer = GlobalBestPSO(n_particles=int(input_pso_particles), dimensions=4, 
                                          options={'c1': 2.0, 'c2': 2.0, 'w': 0.7}, bounds=bounds)
                
                best_cost, best_pos = optimizer.optimize(f_fitness, iters=int(input_pso_iters), 
                                                         X_tr=X_train, y_tr=y_train, X_va=X_test, y_va=y_test, 
                                                         scaler_y=scaler, window=input_window)
                
                # Final Model
                final_model = build_model(best_pos[0], best_pos[1], best_pos[3], input_window)
                final_model.fit(X_train, y_train, epochs=int(input_epochs), batch_size=int(best_pos[2]), verbose=0)
                
                y_p_pso = final_model.predict(X_test)
                mape_pso = mean_absolute_percentage_error(scaler.inverse_transform(y_test), scaler.inverse_transform(y_p_pso)) * 100
                
                st.success(f"Optimasi Berhasil! Units: {int(best_pos[0])}, LR: {best_pos[1]:.4f}")
                st.metric("MAPE GRU-PSO", f"{mape_pso:.4f}%", delta=f"{mape_pso-mape_b:.4f}%")

        with tab3:
            st.subheader("Peramalan 5 Periode ke Depan")
            # Ambil data terakhir untuk input forecast
            last_input = scaled_data[-input_window:].reshape(1, input_window, 1)
            future_raw = forecast_future(final_model, last_input, 5)
            future_real = scaler.inverse_transform(future_raw)
            
            # Buat Tabel Tanggal
            last_date = emas['Tanggal'].max()
            future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
            
            df_forecast = pd.DataFrame({
                'Tanggal': [d.strftime('%d %B %Y') for d in future_dates],
                'Prediksi Harga': [f"Rp {x[0]:,.2f}" for x in future_real]
            })
            st.table(df_forecast)

else:
    st.info("Silakan unggah file Excel untuk memulai.")
