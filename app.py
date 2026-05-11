import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error
)
from pyswarms.single.global_best import GlobalBestPSO
import gc

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="OptimasGRU - Gold Forecasting",
    layout="wide"
)

st.title("📊 Sistem Prediksi Harga Emas (GRU & PSO)")
st.markdown(
    "Aplikasi prediksi harga emas menggunakan "
    "**GRU + Particle Swarm Optimization (PSO)**"
)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("1. Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload File Excel",
    type=["xlsx", "xls"]
)

st.sidebar.caption(
    "Gunakan file dengan kolom: "
    "'Tanggal' dan 'Terakhir'"
)

st.sidebar.header("2. Analisis Data")

show_stat = st.sidebar.checkbox("Statistik Deskriptif")
show_plot = st.sidebar.checkbox("Plot Time Series")
show_missing = st.sidebar.checkbox("Cek Missing Value")

st.sidebar.header("3. Konfigurasi Model")

input_window = st.sidebar.number_input(
    "Window Size (Timestep)",
    min_value=1,
    max_value=30,
    value=3
)

with st.sidebar.expander("⚙️ Pengaturan GRU-PSO"):

    input_pso_particles = st.number_input(
        "Jumlah Partikel PSO",
        min_value=5,
        value=10
    )

    input_pso_iters = st.number_input(
        "Jumlah Iterasi PSO",
        min_value=2,
        value=5
    )

    input_epochs = st.number_input(
        "Epoch Training",
        min_value=5,
        value=30
    )

run_model = st.sidebar.button(
    "🚀 Mulai Training & Prediksi"
)

# ======================================================
# FUNGSI MODEL
# ======================================================
def build_model(units, lr, dropout, window):

    model = Sequential([
        Input(shape=(window, 1)),

        GRU(
            units=int(units),
            activation='tanh'
        ),

        Dropout(dropout),

        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mse'
    )

    return model

# ======================================================
# FITNESS FUNCTION PSO
# ======================================================
def f_fitness(
    particles,
    X_tr,
    y_tr,
    X_va,
    y_va,
    scaler_y,
    window
):

    costs = []

    for p in particles:

        try:
            units = max(1, int(p[0]))
            lr = float(p[1])
            batch = max(1, int(p[2]))
            dropout = float(p[3])

            clear_session()

            model = build_model(
                units,
                lr,
                dropout,
                window
            )

            model.fit(
                X_tr,
                y_tr,
                epochs=5,
                batch_size=batch,
                verbose=0
            )

            y_pred = model.predict(
                X_va,
                verbose=0
            )

            y_pred_inv = scaler_y.inverse_transform(y_pred)

            y_true_inv = scaler_y.inverse_transform(
                y_va.reshape(-1, 1)
            )

            mse = mean_squared_error(
                y_true_inv,
                y_pred_inv
            )

            costs.append(mse)

            del model
            gc.collect()

        except:
            costs.append(999999)

    return np.array(costs)

# ======================================================
# FORECAST FUTURE
# ======================================================
def forecast_future(
    model,
    last_input,
    steps
):

    predictions_scaled = []

    current_input = last_input.copy()

    for _ in range(steps):

        pred = model.predict(
            current_input,
            verbose=0
        )

        predictions_scaled.append(pred[0, 0])

        new_pred = pred.reshape(1, 1, 1)

        if current_input.shape[1] > 1:

            current_input = np.append(
                current_input[:, 1:, :],
                new_pred,
                axis=1
            )

        else:
            current_input = new_pred

    return np.array(predictions_scaled).reshape(-1, 1)

# ======================================================
# PROSES UTAMA
# ======================================================
if uploaded_file is not None:

    try:

        # ==================================================
        # LOAD DATA
        # ==================================================
        emas = pd.read_excel(uploaded_file)

        emas.columns = emas.columns.str.strip()

        st.subheader("📄 Preview Dataset")
        st.write(emas.head())

        # Pastikan kolom tersedia
        required_cols = ['Tanggal', 'Terakhir']

        if not all(col in emas.columns for col in required_cols):

            st.error(
                "Kolom wajib tidak ditemukan.\n"
                "Pastikan ada kolom: "
                "'Tanggal' dan 'Terakhir'"
            )

            st.stop()

        # ==================================================
        # PREPROCESSING
        # ==================================================
        emas = emas[['Tanggal', 'Terakhir']].copy()

        emas.dropna(inplace=True)

        emas['Tanggal'] = pd.to_datetime(
            emas['Tanggal'],
            dayfirst=True,
            errors='coerce'
        )

        emas.dropna(inplace=True)

        emas = emas.sort_values(by='Tanggal')

        # ==================================================
        # ANALISIS DATA
        # ==================================================
        if show_stat:

            st.subheader("📌 Statistik Deskriptif")

            st.write(
                emas['Terakhir'].describe()
            )

        if show_missing:

            st.subheader("📌 Missing Value")

            st.write(
                emas.isnull().sum()
            )

        if show_plot:

            st.subheader("📈 Grafik Harga Emas")

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                emas['Tanggal'],
                emas['Terakhir']
            )

            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Harga")
            ax.set_title("Time Series Harga Emas")

            st.pyplot(fig)

        # ==================================================
        # TRAINING MODEL
        # ==================================================
        if run_model:

            st.divider()

            # ==============================================
            # NORMALISASI
            # ==============================================
            values = emas[['Terakhir']].values

            n_train = int(len(values) * 0.8)

            scaler = MinMaxScaler()

            scaler.fit(values[:n_train])

            scaled_data = scaler.transform(values)

            # ==============================================
            # WINDOWING
            # ==============================================
            X = []
            y = []

            for i in range(input_window, len(scaled_data)):

                X.append(
                    scaled_data[
                        i-input_window:i
                    ]
                )

                y.append(
                    scaled_data[i]
                )

            X = np.array(X)
            y = np.array(y)

            split_idx = n_train - input_window

            X_train = X[:split_idx]
            y_train = y[:split_idx]

            X_test = X[split_idx:]
            y_test = y[split_idx:]

            st.subheader("📊 Shape Data")

            st.write("X_train:", X_train.shape)
            st.write("X_test:", X_test.shape)

            # ==============================================
            # TABS
            # ==============================================
            tab1, tab2, tab3 = st.tabs([
                "Baseline GRU",
                "GRU + PSO",
                "Forecast Future"
            ])

            final_model = None

            # ==============================================
            # BASELINE
            # ==============================================
            with tab1:

                with st.spinner("Training Baseline..."):

                    model_b = build_model(
                        50,
                        0.0001,
                        0.3,
                        input_window
                    )

                    model_b.fit(
                        X_train,
                        y_train,
                        epochs=int(input_epochs),
                        batch_size=32,
                        verbose=0
                    )

                    y_pred_base = model_b.predict(
                        X_test,
                        verbose=0
                    )

                    y_true_inv = scaler.inverse_transform(
                        y_test.reshape(-1, 1)
                    )

                    y_pred_inv = scaler.inverse_transform(
                        y_pred_base
                    )

                    mape_base = (
                        mean_absolute_percentage_error(
                            y_true_inv,
                            y_pred_inv
                        ) * 100
                    )

                    st.metric(
                        "MAPE Baseline",
                        f"{mape_base:.4f}%"
                    )

            # ==============================================
            # GRU + PSO
            # ==============================================
            with tab2:

                with st.spinner(
                    "Optimasi PSO sedang berjalan..."
                ):

                    bounds = (
                        np.array([
                            16,
                            0.0001,
                            16,
                            0.1
                        ]),

                        np.array([
                            128,
                            0.01,
                            128,
                            0.5
                        ])
                    )

                    optimizer = GlobalBestPSO(
                        n_particles=int(
                            input_pso_particles
                        ),

                        dimensions=4,

                        options={
                            'c1': 2.0,
                            'c2': 2.0,
                            'w': 0.7
                        },

                        bounds=bounds
                    )

                    best_cost, best_pos = optimizer.optimize(
                        f_fitness,
                        iters=int(input_pso_iters),

                        X_tr=X_train,
                        y_tr=y_train,

                        X_va=X_test,
                        y_va=y_test,

                        scaler_y=scaler,

                        window=input_window
                    )

                    best_units = int(best_pos[0])
                    best_lr = float(best_pos[1])
                    best_batch = max(1, int(best_pos[2]))
                    best_dropout = float(best_pos[3])

                    final_model = build_model(
                        best_units,
                        best_lr,
                        best_dropout,
                        input_window
                    )

                    final_model.fit(
                        X_train,
                        y_train,
                        epochs=int(input_epochs),
                        batch_size=best_batch,
                        verbose=0
                    )

                    y_pred_pso = final_model.predict(
                        X_test,
                        verbose=0
                    )

                    y_true_inv = scaler.inverse_transform(
                        y_test.reshape(-1, 1)
                    )

                    y_pred_inv = scaler.inverse_transform(
                        y_pred_pso
                    )

                    mape_pso = (
                        mean_absolute_percentage_error(
                            y_true_inv,
                            y_pred_inv
                        ) * 100
                    )

                    st.success("Optimasi berhasil!")

                    st.write(
                        f"""
                        ### Hyperparameter Terbaik
                        - Units : {best_units}
                        - Learning Rate : {best_lr:.5f}
                        - Batch Size : {best_batch}
                        - Dropout : {best_dropout:.3f}
                        """
                    )

                    st.metric(
                        "MAPE GRU-PSO",
                        f"{mape_pso:.4f}%",

                        delta=f"{mape_pso - mape_base:.4f}%"
                    )

            # ==============================================
            # FORECAST
            # ==============================================
            with tab3:

                if final_model is not None:

                    st.subheader(
                        "📈 Prediksi 5 Periode Ke Depan"
                    )

                    last_input = scaled_data[
                        -input_window:
                    ].reshape(
                        1,
                        input_window,
                        1
                    )

                    future_scaled = forecast_future(
                        final_model,
                        last_input,
                        5
                    )

                    future_real = scaler.inverse_transform(
                        future_scaled
                    )

                    last_date = emas['Tanggal'].max()

                    future_dates = [
                        last_date + timedelta(days=i)
                        for i in range(1, 6)
                    ]

                    df_forecast = pd.DataFrame({
                        'Tanggal': [
                            d.strftime('%d %B %Y')
                            for d in future_dates
                        ],

                        'Prediksi Harga': [
                            f"Rp {x[0]:,.2f}"
                            for x in future_real
                        ]
                    })

                    st.table(df_forecast)

                    # ======================================
                    # PLOT FORECAST
                    # ======================================
                    st.subheader("📊 Grafik Forecast")

                    fig2, ax2 = plt.subplots(
                        figsize=(12, 5)
                    )

                    ax2.plot(
                        emas['Tanggal'].tail(30),
                        emas['Terakhir'].tail(30),
                        label='Data Aktual'
                    )

                    ax2.plot(
                        future_dates,
                        future_real.flatten(),
                        marker='o',
                        label='Forecast'
                    )

                    ax2.legend()

                    st.pyplot(fig2)

                else:
                    st.warning(
                        "Model belum berhasil dibuat."
                    )

    except Exception as e:

        st.error(f"Terjadi error: {e}")

else:

    st.info(
        "Silakan upload file Excel terlebih dahulu."
    )
