import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
st.markdown("Aplikasi untuk membandingkan performa model **GRU Standar** dan **GRU-PSO**.")

# --- SIDEBAR ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Data Historis", type=["xlsx", "xls"])

st.sidebar.header("2. Analisis Data")
show_stat = st.sidebar.checkbox("Stat Deskriptif Data")
show_plot = st.sidebar.checkbox("Plot Time Series Data")
show_missing = st.sidebar.checkbox("Cek Missing Value & Outlier")

st.sidebar.header("3. Hyperparameter & Window")
input_window = st.sidebar.number_input("Window Size (Timestep)", min_value=1, max_value=30, value=1)

with st.sidebar.expander("Tuning GRU-PSO"):
    input_lr_range = st.slider("Range Learning Rate", 0.0001, 0.01, (0.0001, 0.01), format="%.4f")
    input_pso_particles = st.number_input("PSO Particles", value=10) # Dikecilkan agar deploy cepat
    input_pso_iters = st.number_input("PSO Iterations", value=5)

run_model = st.sidebar.button("🚀 Mulai Proses Training & Prediksi")

# ==========================================
# 2. FUNGSI LOGIKA (FITNESS FUNCTION PSO)
# ==========================================
def build_model(units, lr, dropout, window):
    model = Sequential([
        Input(shape=(window, 1)),
        GRU(units=int(units), activation='tanh'),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def f_fitness(particles, X_tr, y_tr, X_va, y_va, scaler_y, window):
    costs = []
    for p in particles:
        units, lr, batch, dropout = int(p[0]), p[1], int(p[2]), p[3]
        clear_session()
        model = build_model(units, lr, dropout, window)
        model.fit(X_tr, y_tr, epochs=5, batch_size=batch, verbose=0) # Epoch kecil untuk optimasi
        
        y_pred = model.predict(X_va, verbose=0)
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        y_true_inv = scaler_y.inverse_transform(y_va.reshape(-1,1))
        costs.append(mean_squared_error(y_true_inv, y_pred_inv))
    return np.array(costs)

# ==========================================
# 3. MAIN PROSES
# ==========================================
if uploaded_file is not None:
    emas = pd.read_excel(uploaded_file)
    emas = emas[['Tanggal', 'Terakhir']].dropna()
    emas['Tanggal'] = pd.to_datetime(emas['Tanggal'], dayfirst=True)
    emas = emas.sort_values(by='Tanggal')

    if show_stat:
        st.subheader("📌 Statistik Deskriptif")
        st.write(emas['Terakhir'].describe())

    if show_plot:
        st.subheader("📈 Plot Time Series")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(emas['Tanggal'], emas['Terakhir'], color='#1A5276')
        st.pyplot(fig)

    if run_model:
        st.divider()
        
        # Preprocessing
        values = emas[['Terakhir']].values
        n_train = int(len(values) * 0.8)
        train_data = values[:n_train]
        
        scaler = MinMaxScaler().fit(train_data)
        scaled_values = scaler.transform(values)
        
        # Windowing
        X, y = [], []
        for i in range(input_window, len(scaled_values)):
            X.append(scaled_values[i-input_window:i])
            y.append(scaled_values[i])
        X, y = np.array(X), np.array(y)
        
        split = n_train - input_window
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        tab1, tab2 = st.tabs(["Baseline Model", "GRU-PSO Optimized"])

        # --- BASELINE ---
        with tab1:
            with st.spinner('Training Baseline...'):
                model_b = build_model(50, 0.0001, 0.5, input_window)
                model_b.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
                
                y_p = model_b.predict(X_test)
                y_p_inv = scaler.inverse_transform(y_p)
                y_t_inv = scaler.inverse_transform(y_test)
                
                mape_b = mean_absolute_percentage_error(y_t_inv, y_p_inv) * 100
                st.metric("MAPE Baseline", f"{mape_b:.2f}%")
                
                fig_b, ax_b = plt.subplots()
                ax_b.plot(y_t_inv, label="Aktual")
                ax_b.plot(y_p_inv, label="Prediksi", linestyle="--")
                plt.legend()
                st.pyplot(fig_b)

        # --- GRU-PSO ---
        with tab2:
            with st.spinner('Running PSO Optimization...'):
                # PSO Bounds: [units, lr, batch, dropout]
                lower_b = [16, input_lr_range[0], 16, 0.1]
                upper_b = [128, input_lr_range[1], 128, 0.5]
                bounds = (np.array(lower_b), np.array(upper_b))

                optimizer = GlobalBestPSO(n_particles=int(input_pso_particles), 
                                          dimensions=4, 
                                          options={'c1': 2.0, 'c2': 2.0, 'w': 0.7}, 
                                          bounds=bounds)
                
                # Eksekusi PSO
                best_cost, best_pos = optimizer.optimize(f_fitness, iters=int(input_pso_iters), 
                                                         X_tr=X_train, y_tr=y_train, 
                                                         X_va=X_test, y_va=y_test, 
                                                         scaler_y=scaler, window=input_window)
                
                # Re-train model terbaik
                model_pso = build_model(best_pos[0], best_pos[1], best_pos[3], input_window)
                model_pso.fit(X_train, y_train, epochs=50, batch_size=int(best_pos[2]), verbose=0)
                
                y_p_pso = model_pso.predict(X_test)
                y_p_pso_inv = scaler.inverse_transform(y_p_pso)
                
                mape_pso = mean_absolute_percentage_error(y_t_inv, y_p_pso_inv) * 100
                st.success(f"Best Params: Units={int(best_pos[0])}, LR={best_pos[1]:.4f}")
                st.metric("MAPE GRU-PSO", f"{mape_pso:.2f}%", delta=f"{mape_pso-mape_b:.2f}%")
                
                fig_p, ax_p = plt.subplots()
                ax_p.plot(y_t_inv, label="Aktual")
                ax_p.plot(y_p_pso_inv, label="Prediksi PSO", color="red")
                plt.legend()
                st.pyplot(fig_p)

else:
    st.warning("Upload file Excel dulu di samping!")
