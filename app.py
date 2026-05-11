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

st.sidebar.caption(
    "Format file harus Excel (.xlsx / .xls)"
)

# ======================================================
# SIDEBAR - MENU
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
        "Baseline Model (GRU-Adam)"
    ]
)

# ======================================================
# SIDEBAR - PARAMETER
# ======================================================
st.sidebar.header("⚙️ Parameter Model")

epoch = st.sidebar.number_input(
    "Jumlah Epoch",
    min_value=1,
    value=10
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
    value=16
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
    max_value=3,
    value=1
)

units = st.sidebar.number_input(
    "Jumlah Units",
    min_value=1,
    max_value=128,
    value=16
)

timestep = st.sidebar.number_input(
    "Timestep / Window Size",
    min_value=1,
    max_value=30,
    value=1
)

# ======================================================
# HALAMAN UTAMA
# ======================================================
if uploaded_file is not None:

    try:

        # ==================================================
        # LOAD DATA
        # ==================================================
        df = pd.read_excel(uploaded_file)

        # Bersihkan nama kolom
        df.columns = df.columns.str.strip()

        # Bersihkan missing palsu
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
        # PREVIEW DATASET
        # ==================================================
        if menu == "Preview Dataset":

            st.subheader("📄 Preview Dataset")

            df_preview = df.copy()

            if "Tanggal" in df_preview.columns:

                df_preview["Tanggal"] = pd.to_datetime(
                    df_preview["Tanggal"],
                    errors='coerce'
                ).dt.date

            st.dataframe(
                df_preview.head(),
                use_container_width=True
            )

        # ==================================================
        # STATISTIKA DESKRIPTIF
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

                ax.set_xlabel("Tanggal")

                ax.set_ylabel("Harga")

                ax.grid(alpha=0.3)

                st.pyplot(fig)

            else:

                st.warning(
                    "Kolom 'Tanggal' dan "
                    "'Terakhir' tidak ditemukan."
                )

        # ==================================================
        # MISSING VALUE
        # ==================================================
        elif menu == "Cek Missing Values":

            st.subheader(
                "🧩 Cek Missing Values"
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
                "🚨 Deteksi Outlier (IQR)"
            )

            if "Terakhir" in df.columns:

                Q1 = df["Terakhir"].quantile(0.25)

                Q3 = df["Terakhir"].quantile(0.75)

                IQR = Q3 - Q1

                lower_bound = (
                    Q1 - 1.5 * IQR
                )

                upper_bound = (
                    Q3 + 1.5 * IQR
                )

                outliers = df[
                    (
                        df["Terakhir"]
                        < lower_bound
                    ) |
                    (
                        df["Terakhir"]
                        > upper_bound
                    )
                ]

                outliers = outliers.copy()

                if "Tanggal" in outliers.columns:

                    outliers["Tanggal"] = pd.to_datetime(
                        outliers["Tanggal"],
                        errors='coerce'
                    ).dt.date

                st.write(
                    f"Jumlah Outlier: {len(outliers)}"
                )

                st.dataframe(
                    outliers,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Kolom 'Terakhir' tidak ditemukan."
                )

        # ==================================================
        # BASELINE MODEL
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

                    gc.collect()

                    tf.keras.backend.clear_session()

                    # ======================================
                    # SET SEED
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
                    # BUILD MODEL
                    # ======================================
                    model = Sequential()

                    model.add(
                        Input(
                            shape=(
                                timestep,
                                1
                            )
                        )
                    )

                    # ======================================
                    # SINGLE LAYER
                    # ======================================
                    if layer == 1:

                        model.add(
                            GRU(
                                units=units,
                                activation='tanh'
                            )
                        )

                        model.add(
                            Dropout(dropout)
                        )

                    # ======================================
                    # MULTI LAYER
                    # ======================================
                    else:

                        for i in range(layer):

                            is_last = (
                                i == layer - 1
                            )

                            model.add(
                                GRU(
                                    units=units,
                                    return_sequences=(
                                        not is_last
                                    ),
                                    activation='tanh'
                                )
                            )

                            model.add(
                                Dropout(dropout)
                            )

                    # ======================================
                    # OUTPUT
                    # ======================================
                    model.add(
                        Dense(
                            1,
                            activation='linear'
                        )
                    )

                    # ======================================
                    # COMPILE
                    # ======================================
                    model.compile(
                        optimizer=Adam(
                            learning_rate=
                            learning_rate
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
                        epochs=epoch,
                        batch_size=batch_size,
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
                    # EVALUASI
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
                    # METRICS
                    # ======================================
                    st.subheader(
                        "📊 Hasil Evaluasi"
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "RMSE",
                        f"{rmse:,.2f}"
                    )

                    col2.metric(
                        "MAE",
                        f"{mae:,.2f}"
                    )

                    col3.metric(
                        "MAPE",
                        f"{mape:.4f}%"
                    )

                    # ======================================
                    # LOSS PLOT
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
                    # PREDICT VS ACTUAL
                    # ======================================
                    st.subheader(
                        "📈 Aktual vs Prediksi"
                    )

                    fig2, ax2 = plt.subplots(
                        figsize=(12, 6)
                    )

                    ax2.plot(
                        y_actual,
                        label='Aktual',
                        linewidth=2
                    )

                    ax2.plot(
                        y_pred,
                        label='Prediksi',
                        linestyle='--',
                        linewidth=2
                    )

                    ax2.legend()

                    ax2.grid(alpha=0.3)

                    st.pyplot(fig2)

                    st.success(
                        "Training baseline GRU berhasil!"
                    )

    except Exception as e:

        st.error(
            f"❌ Terjadi error: {e}"
        )

else:

    st.info(
        "📂 Silakan upload file Excel terlebih dahulu."
    )
