import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

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
# SIDEBAR - MENU ANALISIS
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
        "Forecast",
        "Grafik Predict vs Actual"
    ]
)

# ======================================================
# SIDEBAR - PARAMETER
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
        # MEMBACA FILE
        # ==============================================
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

        # ==============================================
        # PREVIEW DATASET
        # ==============================================
        if menu == "Preview Dataset":

            st.subheader("📄 Preview Dataset")

            if "Tanggal" in df.columns:

                df_preview = df.copy()

                df_preview["Tanggal"] = pd.to_datetime(
                    df_preview["Tanggal"],
                    errors='coerce'
                ).dt.date

            else:

                df_preview = df.copy()

            st.dataframe(
                df_preview.head(),
                use_container_width=True
            )

        # ==============================================
        # STATISTIKA DESKRIPTIF
        # ==============================================
        elif menu == "Statistika Deskriptif":

            st.subheader("📊 Statistika Deskriptif")

            numeric_df = df.select_dtypes(
                include=['int64', 'float64']
            )

            st.dataframe(
                numeric_df.describe(),
                use_container_width=True
            )

        # ==============================================
        # TIME SERIES PLOT
        # ==============================================
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

                    df["Terakhir"]
                )

                ax.set_xlabel("Tanggal")
                ax.set_ylabel("Harga")
                ax.set_title(
                    "Time Series Harga Emas"
                )

                st.pyplot(fig)

            else:

                st.warning(
                    "Kolom 'Tanggal' dan "
                    "'Terakhir' tidak ditemukan."
                )

        # ==============================================
        # MISSING VALUE
        # ==============================================
        elif menu == "Cek Missing Values":

            st.subheader(
                "🧩 Cek Missing Values"
            )

            missing_df = pd.DataFrame({
                "Kolom": df.columns,
                "Jumlah Missing":
                df.isnull().sum().values
            })

            st.dataframe(
                missing_df,
                use_container_width=True
            )

        # ==============================================
        # OUTLIER
        # ==============================================
        elif menu == "Cek Outliers":

            st.subheader(
                "🚨 Deteksi Outliers (IQR)"
            )

            if "Terakhir" in df.columns:

                Q1 = df["Terakhir"].quantile(0.25)

                Q3 = df["Terakhir"].quantile(0.75)

                IQR = Q3 - Q1

                lower_bound = (
                    Q1 - (1.5 * IQR)
                )

                upper_bound = (
                    Q3 + (1.5 * IQR)
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

                if "Tanggal" in outliers.columns:

                    outliers = outliers.copy()

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

        # ==============================================
        # BASELINE MODEL
        # ==============================================
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

                    # ==================================
                    # SEED
                    # ==================================
                    SEED = 49

                    random.seed(SEED)

                    np.random.seed(SEED)

                    tf.random.set_seed(SEED)

                    # ==================================
                    # SPLIT DATA
                    # ==================================
                    feature_cols = ["Terakhir"]

                    target_col = "Terakhir"

                    data_features = (
                        df[feature_cols].values
                    )

                    data_target = (
                        df[[target_col]].values
                    )

                    values = (
                        df[['Terakhir']].values
                    )

                    n = len(values)

                    n_train = int(n * 0.8)

                    # ==================================
                    # SCALING
                    # ==================================
                    scaler_X = MinMaxScaler().fit(
                        data_features[:n_train]
                    )

                    scaler_y = MinMaxScaler().fit(
                        data_target[:n_train]
                    )

                    Xs = scaler_X.transform(
                        data_features
                    )

                    ys = scaler_y.transform(
                        data_target
                    )

                    # ==================================
                    # WINDOW
                    # ==================================
                    GS_window = timestep

                    # ==================================
                    # WINDOWING
                    # ==================================
                    def make_sequences(
                        X_scaled,
                        y_scaled,
                        window
                    ):

                        X_seq = []

                        y_seq = []

                        for i in range(
                            window,
                            len(X_scaled)
                        ):

                            X_seq.append(
                                X_scaled[
                                    i-window:i
                                ]
                            )

                            y_seq.append(
                                y_scaled[i]
                            )

                        return (
                            np.array(X_seq),
                            np.array(y_seq)
                        )

                    X_seq_all, y_seq_all = (
                        make_sequences(
                            Xs,
                            ys,
                            window=GS_window
                        )
                    )

                    dtrain_end = (
                        n_train - GS_window
                    )

                    X_train = (
                        X_seq_all[:dtrain_end]
                    )

                    y_train = (
                        y_seq_all[:dtrain_end]
                    )

                    X_test = (
                        X_seq_all[dtrain_end:]
                    )

                    y_test = (
                        y_seq_all[dtrain_end:]
                    )

                    # ==================================
                    # RESHAPE
                    # ==================================
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

                    # ==================================
                    # PARAMETER MODEL
                    # ==================================
                    GS_epoch = epoch

                    GS_batch = batch_size

                    GS_units = units

                    GS_layers = layer

                    GS_dropout = dropout

                    GS_LR = learning_rate

                    # ==================================
                    # BUILD MODEL
                    # ==================================
                    def build_gru_model():

                        model = Sequential()

                        model.add(
                            Input(
                                shape=(
                                    GS_window,
                                    1
                                )
                            )
                        )

                        if GS_layers == 1:

                            model.add(
                                GRU(
                                    units=GS_units,
                                    activation='tanh'
                                )
                            )

                            model.add(
                                Dropout(GS_dropout)
                            )

                        else:

                            for i in range(
                                GS_layers
                            ):

                                is_last = (
                                    i == GS_layers - 1
                                )

                                model.add(
                                    GRU(
                                        units=GS_units,
                                        return_sequences=(
                                            not is_last
                                        ),
                                        activation='tanh'
                                    )
                                )

                                model.add(
                                    Dropout(
                                        GS_dropout
                                    )
                                )

                        model.add(
                            Dense(
                                1,
                                activation='linear'
                            )
                        )

                        model.compile(
                            optimizer=Adam(
                                learning_rate=GS_LR
                            ),
                            loss='mse'
                        )

                        return model

                    # ==================================
                    # TRAINING
                    # ==================================
                    gru_standar = build_gru_model()

                    early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=7,
                        restore_best_weights=True
                    )

                    history = gru_standar.fit(
                        X_train,
                        y_train,
                        epochs=GS_epoch,
                        batch_size=GS_batch,
                        validation_split=0.2,
                        callbacks=[early_stop],
                        verbose=0
                    )

                    # ==================================
                    # PREDIKSI
                    # ==================================
                    y_pred_scaled = (
                        gru_standar.predict(
                            X_test,
                            verbose=0
                        )
                    )

                    y_pred_inv = (
                        scaler_y.inverse_transform(
                            y_pred_scaled
                        ).flatten()
                    )

                    y_test_inv = (
                        scaler_y.inverse_transform(
                            y_test.reshape(-1, 1)
                        ).flatten()
                    )

                    # ==================================
                    # METRICS
                    # ==================================
                    rmse = np.sqrt(
                        mean_squared_error(
                            y_test_inv,
                            y_pred_inv
                        )
                    )

                    mae = mean_absolute_error(
                        y_test_inv,
                        y_pred_inv
                    )

                    mape = (
                        mean_absolute_percentage_error(
                            y_test_inv,
                            y_pred_inv
                        ) * 100
                    )

                    # ==================================
                    # HASIL EVALUASI
                    # ==================================
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

                    # ==================================
                    # PLOT LOSS
                    # ==================================
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

                    # ==================================
                    # PLOT PREDIKSI
                    # ==================================
                    st.subheader(
                        "📈 Aktual vs Prediksi"
                    )

                    fig2, ax2 = plt.subplots(
                        figsize=(12, 6)
                    )

                    ax2.plot(
                        y_test_inv,
                        label='Aktual'
                    )

                    ax2.plot(
                        y_pred_inv,
                        linestyle='--',
                        label='Prediksi'
                    )

                    ax2.legend()

                    ax2.grid(alpha=0.3)

                    st.pyplot(fig2)

                    st.success(
                        "Training baseline GRU berhasil!"
                    )

        # ==============================================
        # FORECAST
        # ==============================================
        elif menu == "Forecast":

            st.subheader("🔮 Forecast")

            st.info(
                "Forecast akan ditambahkan berikutnya."
            )

        # ==============================================
        # PREDICT VS ACTUAL
        # ==============================================
        elif menu == "Grafik Predict vs Actual":

            st.subheader(
                "📉 Grafik Predict vs Actual"
            )

            st.info(
                "Grafik predict vs actual akan ditambahkan berikutnya."
            )

    except Exception as e:

        st.error(
            f"❌ Terjadi error: {e}"
        )

else:

    st.info(
        "📂 Silakan upload file Excel terlebih dahulu."
    )
