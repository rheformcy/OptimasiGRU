import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gc

# ======================================================
# LIBRARIES
# ======================================================
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
import pyswarms as ps

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(page_title="Optimasi GRU", layout="wide")

st.markdown("""
# 📊 Optimasi Gated Recurrent Unit (GRU)
**Rhena Amelia Shafitry** | Statistika UNDIP | 24050122120019
""")
st.divider()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("📂 Data & Menu")
uploaded_file = st.sidebar.file_uploader("Upload File Excel", type=["xlsx", "xls"])

menu = st.sidebar.radio(
    "Pilih Menu",
    ["Preview Dataset", "Statistika Deskriptif", "Visualisasi Plot", "Cek Missing & Outliers", "Baseline Model", "Optimasi GRU-PSO"]
)

st.sidebar.divider()
st.sidebar.header("⚙️ Hyperparameters")
timestep = st.sidebar.number_input("Timestep (Window)", min_value=1, value=1)
particle = st.sidebar.number_input("Partikel PSO", min_value=1, value=5)
iterasi = st.sidebar.number_input("Iterasi PSO", min_value=1, value=3)
epoch_final = st.sidebar.number_input("Epoch Final", min_value=1, value=30)

# ======================================================
# MAIN LOGIC
# ======================================================
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    
    # Bersihkan Data
    df = df.replace([r'^\s*$', "-", "?", "null", "NULL"], pd.NA, regex=True)
    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"]).dt.date

    # 1. PREVIEW
    if menu == "Preview Dataset":
        st.subheader("📄 Cuplikan Data")
        st.dataframe(df.head(10), height=250)

    # 2. DESKRIPTIF
    elif menu == "Statistika Deskriptif":
        st.subheader("📊 Ringkasan Statistik")
        st.dataframe(df.describe(), use_container_width=True)

    # 3. VISUALISASI
    elif menu == "Visualisasi Plot":
        st.subheader("📈 Time Series Plot")
        if "Terakhir" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 3)) # Diperkecil
            ax.plot(df["Tanggal"], df["Terakhir"], color='#1f77b4')
            ax.set_title("Tren Harga")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

    # 4. MISSING & OUTLIERS
    elif menu == "Cek Missing & Outliers":
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())
        with col2:
            if "Terakhir" in df.columns:
                Q1, Q3 = df["Terakhir"].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                outliers = df[(df["Terakhir"] < (Q1 - 1.5*IQR)) | (df["Terakhir"] > (Q3 + 1.5*IQR))]
                st.write(f"**Jumlah Outlier:** {len(outliers)}")
                st.dataframe(outliers, height=200)

    # 5. BASELINE MODEL
    elif menu == "Baseline Model":
        st.subheader("🤖 Baseline Model (GRU-Adam)")
        
        # Tabel Konfigurasi Baseline
        config_data = {
            "Parameter": ["Epoch", "Batch Size", "Units", "Dropout", "Learning Rate", "Timestep"],
            "Value": [100, 64, 50, 0.2, 0.0001, timestep]
        }
        st.table(pd.DataFrame(config_data))

        if st.button("Jalankan Baseline"):
            with st.spinner("Training..."):
                clear_session()
                # Data Preparation
                values = df[['Terakhir']].values.astype('float32')
                n_train = int(len(values) * 0.8)
                scaler = MinMaxScaler().fit(values[:n_train])
                scaled = scaler.transform(values)

                X, y = [], []
                for i in range(timestep, len(scaled)):
                    X.append(scaled[i-timestep:i])
                    y.append(scaled[i])
                X, y = np.array(X), np.array(y)

                X_train, X_test = X[:n_train-timestep], X[n_train-timestep:]
                y_train, y_test = y[:n_train-timestep], y[n_train-timestep:]

                # Model
                model = Sequential([
                    Input(shape=(timestep, 1)),
                    GRU(50, activation='tanh'),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer=Adam(0.0001), loss='mse')
                history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, 
                                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)

                # Metrics
                preds = scaler.inverse_transform(model.predict(X_test))
                actual = scaler.inverse_transform(y_test)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("RMSE", f"{np.sqrt(mean_squared_error(actual, preds)):.2f}")
                c2.metric("MAE", f"{mean_absolute_error(actual, preds):.2f}")
                c3.metric("MAPE", f"{mean_absolute_percentage_error(actual, preds)*100:.2f}%")

                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(actual, label="Actual")
                ax.plot(preds, label="Pred")
                ax.legend(); st.pyplot(fig)

    # 6. OPTIMASI PSO
    elif menu == "Optimasi GRU-PSO":
        st.subheader("🚀 Optimasi GRU-PSO")
        
        # Tampilkan parameter yang akan digunakan
        st.info(f"Target: Mencari Unit, LR, Batch, dan Dropout optimal dengan {particle} Partikel & {iterasi} Iterasi.")
        
        if st.button("Mulai Optimasi GRU-PSO"):
            progress_bar = st.progress(0)
            with st.spinner("PSO sedang mencari parameter terbaik..."):
                # (Logika PSO diringkas untuk efisiensi ruang)
                values = df[['Terakhir']].values.astype('float32')
                n_train = int(len(values) * 0.8)
                scaler = MinMaxScaler().fit(values[:n_train])
                scaled = scaler.transform(values)

                # Windowing
                X, y = [], []
                for i in range(timestep, len(scaled)):
                    X.append(scaled[i-timestep:i]); y.append(scaled[i])
                X, y = np.array(X), np.array(y)
                X_tr = X[:int(len(X)*0.8)]; y_tr = y[:int(len(y)*0.8)]
                X_val = X[int(len(X)*0.8):]; y_val = y[int(len(y)*0.8):]

                def fitness_func(particles):
                    results = []
                    for p in particles:
                        try:
                            clear_session()
                            m = Sequential([Input(shape=(timestep, 1)), GRU(int(p[0])), Dropout(p[3]), Dense(1)])
                            m.compile(optimizer=Adam(p[1]), loss='mse')
                            m.fit(X_tr, y_tr, epochs=5, batch_size=int(p[2]), verbose=0)
                            err = mean_squared_error(y_val, m.predict(X_val, verbose=0))
                            results.append(err)
                        except: results.append(999)
                    return np.array(results)

                bounds = ([8, 0.0001, 8, 0.0], [128, 0.01, 128, 0.5])
                opt = ps.single.GlobalBestPSO(n_particles=particle, dimensions=4, options={'c1':2, 'c2':2, 'w':0.7}, bounds=bounds)
                best_cost, best_pos = opt.optimize(fitness_func, iters=iterasi)

                st.success("Optimasi Selesai!")
                res_df = pd.DataFrame([best_pos], columns=["Units", "LR", "Batch", "Dropout"])
                st.write("**Parameter Terbaik Found:**")
                st.table(res_df)

                # Final Plot Convergence
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.plot(opt.cost_history)
                ax.set_title("Konvergensi PSO")
                st.pyplot(fig)

else:
    st.info("Silakan unggah file Excel terlebih dahulu melalui sidebar.")
