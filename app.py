import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gc
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session

import pyswarms as ps

# ======================================================
# CONFIG
# ======================================================
tf.keras.backend.clear_session()
gc.collect()

st.set_page_config(
    page_title="Optimasi GRU",
    layout="wide"
)

# ======================================================
# HEADER
# ======================================================
st.markdown("""
# 📊 Optimasi GRU (GRU-PSO)

### Rhena Amelia Shafitry  
Statistika UNDIP  
24050122120019
""")

st.divider()

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx", "xls"])

st.sidebar.header("📌 Menu")
menu = st.sidebar.radio("Pilih Menu", [
    "Preview Dataset",
    "Statistika Deskriptif",
    "Visualisasi Time Series Plot",
    "Cek Missing Values",
    "Cek Outliers",
    "Baseline Model (GRU-Adam)",
    "Optimasi GRU-PSO"
])

# ======================================================
# PARAMETER
# ======================================================
st.sidebar.header("⚙️ Parameter")

timestep = st.sidebar.number_input("Timestep", 1, 30, 1)

particle = st.sidebar.number_input("Partikel", 1, value=5)
iterasi = st.sidebar.number_input("Iterasi PSO", 1, value=3)
epoch_final = st.sidebar.number_input("Epoch Final", 1, value=30)

units_min = st.sidebar.number_input("Units Min", 1, 8)
units_max = st.sidebar.number_input("Units Max", 1, 128)

lr_min = st.sidebar.number_input("LR Min", 0.0001, value=0.0001, step=0.0001, format="%.4f")
lr_max = st.sidebar.number_input("LR Max", 0.0001, value=0.01, step=0.0001, format="%.4f")

batch_min = st.sidebar.number_input("Batch Min", 1, 8)
batch_max = st.sidebar.number_input("Batch Max", 1, 128)

dropout_min = st.sidebar.slider("Dropout Min", 0.0, 0.9, 0.0, 0.1)
dropout_max = st.sidebar.slider("Dropout Max", 0.0, 0.9, 0.5, 0.1)

# ======================================================
# LOAD DATA
# ======================================================
if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    df.replace(["-", "?", "null", "NULL"], pd.NA, inplace=True)

    # ==================================================
    # PREVIEW
    # ==================================================
    if menu == "Preview Dataset":
        st.subheader("📄 Preview")
        st.dataframe(df.head(), use_container_width=True)

    # ==================================================
    # DESKRIPTIF
    # ==================================================
    elif menu == "Statistika Deskriptif":
        st.subheader("📊 Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)

    # ==================================================
    # TIME SERIES
    # ==================================================
    elif menu == "Visualisasi Time Series Plot":
        st.subheader("📈 Time Series")

        if "Tanggal" in df.columns and "Terakhir" in df.columns:

            fig, ax = plt.subplots(figsize=(7, 3))

            ax.plot(
                pd.to_datetime(df["Tanggal"]).dt.date,
                df["Terakhir"],
                linewidth=1.5
            )

            ax.set_title("Harga Emas")
            ax.grid(alpha=0.3)

            st.pyplot(fig)

    # ==================================================
    # MISSING
    # ==================================================
    elif menu == "Cek Missing Values":

        st.subheader("🧩 Missing Values")

        miss = pd.DataFrame({
            "Kolom": df.columns,
            "Missing": df.isnull().sum().values
        })

        st.dataframe(miss, use_container_width=True)

    # ==================================================
    # OUTLIER
    # ==================================================
    elif menu == "Cek Outliers":

        st.subheader("🚨 Outlier")

        if "Terakhir" in df.columns:

            Q1 = df["Terakhir"].quantile(0.25)
            Q3 = df["Terakhir"].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            out = df[(df["Terakhir"] < lower) | (df["Terakhir"] > upper)]

            st.write(f"Outlier: {len(out)}")
            st.dataframe(out, use_container_width=True)

    # ======================================================
    # BASELINE
    # ======================================================
    elif menu == "Baseline Model (GRU-Adam)":

        st.subheader("🤖 Baseline GRU")

        config_df = pd.DataFrame({
            "Parameter": ["Epoch", "Batch", "Units", "Dropout", "LR", "Timestep"],
            "Nilai": [100, 64, 50, 0.2, 0.0001, timestep]
        })

        st.markdown("### ⚙️ Konfigurasi Baseline")
        st.dataframe(config_df, use_container_width=False)

        st.info("Model baseline dijalankan saat training manual (tidak otomatis).")

    # ======================================================
    # PSO
    # ======================================================
    elif menu == "Optimasi GRU-PSO":

        st.subheader("🚀 GRU-PSO")

        run = st.button("▶️ Mulai Optimasi GRU-PSO")

        if run:

            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = time.time()

            clear_session()
            gc.collect()

            # ================= DATA =================
            values = df[['Terakhir']].values
            n_train = int(len(values) * 0.8)

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(values)

            X, y = [], []

            for i in range(timestep, len(scaled)):
                X.append(scaled[i-timestep:i])
                y.append(scaled[i])

            X = np.array(X)
            y = np.array(y)

            split = n_train - timestep

            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            X_train = X_train.reshape(X_train.shape[0], timestep, 1)
            X_test = X_test.reshape(X_test.shape[0], timestep, 1)

            val_size = int(len(X_train) * 0.2)
            X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
            y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

            # ================= OBJECTIVE =================
            def objective(particles):

                losses = np.zeros(particles.shape[0])

                for i, p in enumerate(particles):

                    try:
                        units = int(np.round(p[0]))
                        lr = float(p[1])
                        batch = int(np.round(p[2]))
                        dropout = float(p[3])

                        clear_session()

                        model = Sequential([
                            Input(shape=(timestep, 1)),
                            GRU(units),
                            Dropout(dropout),
                            Dense(1)
                        ])

                        model.compile(Adam(lr), loss="mse")

                        model.fit(X_tr, y_tr, epochs=5, batch_size=batch, verbose=0)

                        pred = model.predict(X_val, verbose=0)

                        losses[i] = mean_squared_error(
                            scaler.inverse_transform(y_val),
                            scaler.inverse_transform(pred)
                        )

                    except:
                        losses[i] = 1e9

                return losses

            # ================= PSO =================
            optimizer = ps.single.GlobalBestPSO(
                n_particles=particle,
                dimensions=4,
                options={'c1': 2, 'c2': 2, 'w': 0.7},
                bounds=(
                    [units_min, lr_min, batch_min, dropout_min],
                    [units_max, lr_max, batch_max, dropout_max]
                )
            )

            # ================= PROGRESS LOOP =================
            for i in range(iterasi):

                cost, pos = optimizer.optimize(
                    objective,
                    iters=1,
                    verbose=False
                )

                percent = int((i + 1) / iterasi * 100)

                progress_bar.progress(percent / 100)

                elapsed = time.time() - start_time

                status_text.write(
                    f"🔄 Iterasi {i+1}/{iterasi} | {percent}% | {elapsed:.1f}s"
                )

            progress_bar.progress(1.0)
            status_text.success("✅ Optimasi selesai!")
