

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import gc

# ======================================================
# SCIKIT LEARN
# ======================================================
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

# ======================================================
# TENSORFLOW
# ======================================================
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    GRU,
    Dense,
    Dropout,
    Input
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from tensorflow.keras.backend import clear_session

# ======================================================
# PSO
# ======================================================
import pyswarms as ps

# ======================================================
# CLEAR SESSION
# ======================================================
tf.keras.backend.clear_session()
gc.collect()

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
# SIDEBAR - UPLOAD FILE
# ======================================================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload File Excel",
    type=["xlsx", "xls"]
)

# ======================================================
# MENU
# ======================================================
st.sidebar.header("📌 Pilihan Analisis")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Preview Dataset",
        "Statistika Deskriptif",
        "Visualisasi Time Series Plot",
        "Cek Missing Values",
        "Cek Outliers",
        "Baseline Model (GRU-Adam)",
        "Optimasi GRU-PSO"
    ]
)

# ======================================================
# PARAMETER BASELINE
# ======================================================
st.sidebar.header("⚙️ Parameter Baseline")

timestep = st.sidebar.number_input(
    "Timestep / Window Size",
    min_value=1,
    max_value=30,
    value=1
)

# ======================================================
# PARAMETER PSO
# ======================================================
st.sidebar.header("🚀 Parameter PSO")

particle = st.sidebar.number_input(
    "Jumlah Partikel",
    min_value=1,
    value=5
)

iterasi = st.sidebar.number_input(
    "Jumlah Iterasi",
    min_value=1,
    value=3
)

epoch = st.sidebar.number_input(
    "Epoch Final Training",
    min_value=1,
    value=30
)

layer = st.sidebar.number_input(
    "Jumlah Layer",
    min_value=1,
    max_value=3,
    value=1
)

# ======================================================
# RANGE UNITS
# ======================================================
st.sidebar.subheader("Range Units")

units_min = st.sidebar.number_input(
    "Units Minimum",
    min_value=1,
    value=8
)

units_max = st.sidebar.number_input(
    "Units Maximum",
    min_value=1,
    value=128
)

# ======================================================
# RANGE LEARNING RATE
# ======================================================
st.sidebar.subheader("Range Learning Rate")

lr_min = st.sidebar.number_input(
    "LR Minimum",
    min_value=0.0001,
    value=0.0001,
    step=0.0001,
    format="%.4f"
)

lr_max = st.sidebar.number_input(
    "LR Maximum",
    min_value=0.0001,
    value=0.01,
    step=0.0001,
    format="%.4f"
)

# ======================================================
# RANGE BATCH
# ======================================================
st.sidebar.subheader("Range Batch Size")

batch_min = st.sidebar.number_input(
    "Batch Minimum",
    min_value=1,
    value=8
)

batch_max = st.sidebar.number_input(
    "Batch Maximum",
    min_value=1,
    value=128
)

# ======================================================
# RANGE DROPOUT
# ======================================================
st.sidebar.subheader("Range Dropout")

dropout_min = st.sidebar.slider(
    "Dropout Minimum",
    min_value=0.0,
    max_value=0.9,
    value=0.0,
    step=0.1
)

dropout_max = st.sidebar.slider(
    "Dropout Maximum",
    min_value=0.0,
    max_value=0.9,
    value=0.5,
    step=0.1
)

# ======================================================
# MAIN
# ======================================================
if uploaded_file is not None:

    try:

        # ==================================================
        # LOAD DATA
        # ==================================================
        df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.strip()

        df = df.replace(
            r'^\s*$',
            pd.NA,
            regex=True
        )

        df.replace(
            ["-", "?", "null", "NULL"],
            pd.NA,
            inplace=True
        )

        # ==================================================
        # PREVIEW
        # ==================================================
        if menu == "Preview Dataset":

            st.subheader("📄 Preview Dataset")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

        # ==================================================
        # DESKRIPTIF
        # ==================================================
        elif menu == "Statistika Deskriptif":

            st.subheader(
                "📊 Statistika Deskriptif"
            )

            numeric_df = df.select_dtypes(
                include=['int64', 'float64']
            )

            st.dataframe(
                numeric_df.describe(),
                use_container_width=True
            )

        # ==================================================
        # VISUALISASI
        # ==================================================
        elif menu == "Visualisasi Time Series Plot":

            st.subheader(
                "📈 Visualisasi Time Series Plot"
            )

            if (
                "Tanggal" in df.columns and
                "Terakhir" in df.columns
            ):

                fig, ax = plt.subplots(
                    figsize=(12, 5)
                )

                ax.plot(
                    pd.to_datetime(
                        df["Tanggal"]
                    ).dt.date,

                    df["Terakhir"],
                    linewidth=2
                )

                ax.set_title(
                    "Time Series Harga Emas"
                )

                ax.grid(alpha=0.3)

                st.pyplot(fig)

        # ==================================================
        # MISSING VALUE
        # ==================================================
        elif menu == "Cek Missing Values":

            st.subheader(
                "🧩 Missing Values"
            )

            missing_df = pd.DataFrame({

                "Kolom":
                df.columns,

                "Jumlah Missing":
                df.isnull().sum().values
            })

            st.dataframe(
                missing_df,
                use_container_width=True
            )

        # ==================================================
        # OUTLIER
        # ==================================================
        elif menu == "Cek Outliers":

            st.subheader(
                "🚨 Deteksi Outlier"
            )

            if "Terakhir" in df.columns:

                Q1 = df["Terakhir"].quantile(0.25)

                Q3 = df["Terakhir"].quantile(0.75)

                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR

                upper = Q3 + 1.5 * IQR

                outliers = df[
                    (
                        df["Terakhir"] < lower
                    ) |
                    (
                        df["Terakhir"] > upper
                    )
                ]

                st.write(
                    f"Jumlah Outlier: {len(outliers)}"
                )

                st.dataframe(
                    outliers,
                    use_container_width=True
                )

        # ==================================================
        # BASELINE GRU
        # ==================================================
        elif menu == "Baseline Model (GRU-Adam)":

            st.subheader(
                "🤖 Baseline Model (GRU-Adam)"
            )

            if "Terakhir" not in df.columns:

                st.warning(
                    "Kolom 'Terakhir' tidak ditemukan."
                )

            else:

                with st.spinner(
                    "Training baseline GRU..."
                ):

                    clear_session()

                    gc.collect()

                    # ======================================
                    # FIXED PARAMETER
                    # ======================================
                    GS_epoch = 100
                    GS_batch = 64
                    GS_units = 50
                    GS_layers = 1
                    GS_dropout = 0.2
                    GS_lr = 0.0001
                    GS_window = timestep

                    # ======================================
                    # SEED
                    # ======================================
                    SEED = 49

                    random.seed(SEED)

                    np.random.seed(SEED)

                    tf.random.set_seed(SEED)

                    # ======================================
                    # DATA
                    # ======================================
                    values = df[
                        ['Terakhir']
                    ].values

                    n = len(values)

                    n_train = int(n * 0.8)

                    # ======================================
                    # SCALING
                    # ======================================
                    scaler = MinMaxScaler()

                    scaler.fit(
                        values[:n_train]
                    )

                    scaled_data = scaler.transform(
                        values
                    )

                    # ======================================
                    # WINDOWING
                    # ======================================
                    X = []

                    y = []

                    for i in range(
                        GS_window,
                        len(scaled_data)
                    ):

                        X.append(
                            scaled_data[
                                i-GS_window:i
                            ]
                        )

                        y.append(
                            scaled_data[i]
                        )

                    X = np.array(X)

                    y = np.array(y)

                    split_idx = (
                        n_train - GS_window
                    )

                    X_train = X[:split_idx]

                    y_train = y[:split_idx]

                    X_test = X[split_idx:]

                    y_test = y[split_idx:]

                    # ======================================
                    # RESHAPE
                    # ======================================
                    X_train = X_train.reshape(
                        (
                            X_train.shape[0],
                            X_train.shape[1],
                            1
                        )
                    )

                    X_test = X_test.reshape(
                        (
                            X_test.shape[0],
                            X_test.shape[1],
                            1
                        )
                    )

                    # ======================================
                    # MODEL
                    # ======================================
                    model = Sequential()

                    model.add(
                        Input(
                            shape=(
                                GS_window,
                                1
                            )
                        )
                    )

                    model.add(
                        GRU(
                            units=GS_units,
                            activation='tanh'
                        )
                    )

                    model.add(
                        Dropout(GS_dropout)
                    )

                    model.add(Dense(1))

                    # ======================================
                    # COMPILE
                    # ======================================
                    model.compile(
                        optimizer=Adam(
                            learning_rate=GS_lr
                        ),
                        loss='mse'
                    )

                    # ======================================
                    # EARLY STOPPING
                    # ======================================
                    early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )

                    # ======================================
                    # TRAINING
                    # ======================================
                    history = model.fit(
                        X_train,
                        y_train,
                        epochs=GS_epoch,
                        batch_size=GS_batch,
                        validation_split=0.2,
                        callbacks=[early_stop],
                        verbose=1
                    )

                    # ======================================
                    # PREDIKSI
                    # ======================================
                    y_pred_scaled = model.predict(
                        X_test,
                        verbose=0
                    )

                    y_pred = scaler.inverse_transform(
                        y_pred_scaled
                    ).flatten()

                    y_actual = scaler.inverse_transform(
                        y_test.reshape(-1, 1)
                    ).flatten()

                    # ======================================
                    # METRICS
                    # ======================================
                    rmse = np.sqrt(
                        mean_squared_error(
                            y_actual,
                            y_pred
                        )
                    )

                    mae = mean_absolute_error(
                        y_actual,
                        y_pred
                    )

                    mape = (
                        mean_absolute_percentage_error(
                            y_actual,
                            y_pred
                        ) * 100
                    )

                    # ======================================
                    # HASIL
                    # ======================================
                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "RMSE",
                        f"Rp {rmse:,.2f}"
                    )

                    col2.metric(
                        "MAE",
                        f"Rp {mae:,.2f}"
                    )

                    col3.metric(
                        "MAPE",
                        f"{mape:.4f}%"
                    )

                    # ======================================
                    # LOSS
                    # ======================================
                    st.subheader(
                        "📉 Training vs Validation Loss"
                    )

                    fig1, ax1 = plt.subplots(
                        figsize=(10, 5)
                    )

                    ax1.plot(
                        history.history['loss'],
                        label='Training Loss'
                    )

                    ax1.plot(
                        history.history['val_loss'],
                        label='Validation Loss'
                    )

                    ax1.legend()

                    ax1.grid(alpha=0.3)

                    st.pyplot(fig1)

                    # ======================================
                    # AKTUAL VS PREDIKSI
                    # ======================================
                    st.subheader(
                        "📈 Aktual vs Prediksi"
                    )

                    fig2, ax2 = plt.subplots(
                        figsize=(12, 6)
                    )

                    ax2.plot(
                        y_actual,
                        label='Aktual'
                    )

                    ax2.plot(
                        y_pred,
                        label='Prediksi'
                    )

                    ax2.legend()

                    ax2.grid(alpha=0.3)

                    st.pyplot(fig2)

                    st.success(
                        "Training baseline berhasil!"
                    )

        # ==================================================
        # OPTIMASI GRU-PSO
        # ==================================================
        elif menu == "Optimasi GRU-PSO":

            st.subheader(
                "🚀 Optimasi GRU-PSO"
            )

            with st.spinner(
                "Proses optimasi..."
            ):

                clear_session()

                gc.collect()

                # ======================================
                # SEED
                # ======================================
                SEED = 987

                random.seed(SEED)

                np.random.seed(SEED)

                tf.random.set_seed(SEED)

                # ======================================
                # DATA
                # ======================================
                values = df[
                    ['Terakhir']
                ].values

                n = len(values)

                n_train = int(n * 0.8)

                # ======================================
                # SCALING
                # ======================================
                scaler = MinMaxScaler()

                scaler.fit(values[:n_train])

                scaled_data = scaler.transform(values)

                # ======================================
                # WINDOWING
                # ======================================
                X = []

                y = []

                for i in range(
                    timestep,
                    len(scaled_data)
                ):

                    X.append(
                        scaled_data[
                            i-timestep:i
                        ]
                    )

                    y.append(
                        scaled_data[i]
                    )

                X = np.array(X)

                y = np.array(y)

                split_idx = (
                    n_train - timestep
                )

                X_train = X[:split_idx]

                y_train = y[:split_idx]

                X_test = X[split_idx:]

                y_test = y[split_idx:]

                # ======================================
                # RESHAPE
                # ======================================
                X_train = X_train.reshape(
                    (
                        X_train.shape[0],
                        X_train.shape[1],
                        1
                    )
                )

                X_test = X_test.reshape(
                    (
                        X_test.shape[0],
                        X_test.shape[1],
                        1
                    )
                )

                # ======================================
                # VALIDATION
                # ======================================
                val_size = 0.2

                train_size = int(
                    len(X_train) * (1 - val_size)
                )

                X_tr = X_train[:train_size]

                y_tr = y_train[:train_size]

                X_val = X_train[train_size:]

                y_val = y_train[train_size:]

                # ======================================
                # PSO OPTIONS
                # ======================================
                options = {
                    'c1': 2.0,
                    'c2': 2.0,
                    'w': 0.7
                }

                bounds = (
                    [
                        units_min,
                        lr_min,
                        batch_min,
                        dropout_min
                    ],

                    [
                        units_max,
                        lr_max,
                        batch_max,
                        dropout_max
                    ]
                )

                # ======================================
                # FITNESS FUNCTION
                # ======================================
                def objective_function(
                    particles
                ):

                    n_particles = (
                        particles.shape[0]
                    )

                    losses = np.zeros(
                        n_particles
                    )

                    for i, particle_i in enumerate(
                        particles
                    ):

                        try:

                            units_p = int(
                                np.round(
                                    particle_i[0]
                                )
                            )

                            lr_p = float(
                                particle_i[1]
                            )

                            batch_p = int(
                                np.round(
                                    particle_i[2]
                                )
                            )

                            dropout_p = float(
                                particle_i[3]
                            )

                            clear_session()

                            model = Sequential()

                            model.add(
                                Input(
                                    shape=(
                                        timestep,
                                        1
                                    )
                                )
                            )

                            model.add(
                                GRU(
                                    units=units_p,
                                    activation='tanh'
                                )
                            )

                            model.add(
                                Dropout(
                                    dropout_p
                                )
                            )

                            model.add(Dense(1))

                            model.compile(
                                optimizer=Adam(
                                    learning_rate=lr_p
                                ),
                                loss='mse'
                            )

                            model.fit(
                                X_tr,
                                y_tr,
                                epochs=5,
                                batch_size=batch_p,
                                verbose=0
                            )

                            pred = model.predict(
                                X_val,
                                verbose=0
                            )

                            pred_inv = scaler.inverse_transform(
                                pred
                            ).flatten()

                            actual_inv = scaler.inverse_transform(
                                y_val.reshape(-1, 1)
                            ).flatten()

                            mse = mean_squared_error(
                                actual_inv,
                                pred_inv
                            )

                            losses[i] = mse

                            clear_session()

                            gc.collect()

                        except:

                            losses[i] = 999999999

                    return losses

                # ======================================
                # RUN PSO
                # ======================================
                optimizer = ps.single.GlobalBestPSO(
                    n_particles=particle,
                    dimensions=4,
                    options=options,
                    bounds=bounds
                )

                best_cost, best_pos = optimizer.optimize(
                    objective_function,
                    iters=iterasi
                )

                # ======================================
                # BEST PARAMETER
                # ======================================
                best_units = int(
                    np.round(best_pos[0])
                )

                best_lr = float(
                    best_pos[1]
                )

                best_batch = int(
                    np.round(best_pos[2])
                )

                best_dropout = float(
                    best_pos[3]
                )

                # ======================================
                # FINAL MODEL
                # ======================================
                clear_session()

                model_final = Sequential()

                model_final.add(
                    Input(
                        shape=(
                            timestep,
                            1
                        )
                    )
                )

                model_final.add(
                    GRU(
                        units=best_units,
                        activation='tanh'
                    )
                )

                model_final.add(
                    Dropout(best_dropout)
                )

                model_final.add(Dense(1))

                model_final.compile(
                    optimizer=Adam(
                        learning_rate=best_lr
                    ),
                    loss='mse'
                )

                history = model_final.fit(
                    X_train,
                    y_train,
                    epochs=epoch,
                    batch_size=best_batch,
                    validation_split=0.2,
                    verbose=1
                )

                # ======================================
                # PREDIKSI
                # ======================================
                y_pred_scaled = model_final.predict(
                    X_test,
                    verbose=0
                )

                y_pred = scaler.inverse_transform(
                    y_pred_scaled
                ).flatten()

                y_actual = scaler.inverse_transform(
                    y_test.reshape(-1, 1)
                ).flatten()

                # ======================================
                # METRICS
                # ======================================
                rmse = np.sqrt(
                    mean_squared_error(
                        y_actual,
                        y_pred
                    )
                )

                mae = mean_absolute_error(
                    y_actual,
                    y_pred
                )

                mape = (
                    mean_absolute_percentage_error(
                        y_actual,
                        y_pred
                    ) * 100
                )

                # ======================================
                # BEST PARAMETER
                # ======================================
                st.subheader(
                    "🏆 Best Hyperparameter"
                )

                best_df = pd.DataFrame({

                    "Units":
                    [best_units],

                    "Learning Rate":
                    [best_lr],

                    "Batch Size":
                    [best_batch],

                    "Dropout":
                    [best_dropout]
                })

                st.dataframe(
                    best_df,
                    use_container_width=True
                )

                # ======================================
                # EVALUASI
                # ======================================
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "RMSE",
                    f"Rp {rmse:,.2f}"
                )

                col2.metric(
                    "MAE",
                    f"Rp {mae:,.2f}"
                )

                col3.metric(
                    "MAPE",
                    f"{mape:.4f}%"
                )

                # ======================================
                # KONVERGENSI
                # ======================================
                st.subheader(
                    "📉 Grafik Konvergensi PSO"
                )

                fig1, ax1 = plt.subplots(
                    figsize=(10, 5)
                )

                ax1.plot(
                    optimizer.cost_history,
                    marker='o'
                )

                ax1.set_xlabel(
                    "Iterasi"
                )

                ax1.set_ylabel(
                    "Best MSE"
                )

                ax1.grid(alpha=0.3)

                st.pyplot(fig1)

                # ======================================
                # LOSS
                # ======================================
                st.subheader(
                    "📉 Training vs Validation Loss"
                )

                fig2, ax2 = plt.subplots(
                    figsize=(10, 5)
                )

                ax2.plot(
                    history.history['loss'],
                    label='Training Loss'
                )

                ax2.plot(
                    history.history['val_loss'],
                    label='Validation Loss'
                )

                ax2.legend()

                ax2.grid(alpha=0.3)

                st.pyplot(fig2)

                # ======================================
                # AKTUAL VS PREDIKSI
                # ======================================
                st.subheader(
                    "📈 Aktual vs Prediksi"
                )

                fig3, ax3 = plt.subplots(
                    figsize=(12, 6)
                )

                ax3.plot(
                    y_actual,
                    label='Aktual'
                )

                ax3.plot(
                    y_pred,
                    label='Prediksi'
                )

                ax3.legend()

                ax3.grid(alpha=0.3)

                st.pyplot(fig3)

                st.success(
                    "Optimasi GRU-PSO berhasil!"
                )

    except Exception as e:

        st.error(
            f"❌ Terjadi error: {e}"
        )

else:

    st.info(
        "📂 Silakan upload file Excel terlebih dahulu."
    )
