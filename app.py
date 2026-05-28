import os
import gc
import random
import warnings

warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================
# SEED
# ==========================================
SEED_VALUE = 49
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

from pyswarms.single.global_best import GlobalBestPSO

# =====================================================
# CONFIG PAGE
# =====================================================
st.set_page_config(
    page_title="GRU-PSO Gold Forecasting",
    layout="wide"
)

st.title("GRU-PSO Forecasting Harga Emas")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("Konfigurasi Model")

window = st.sidebar.number_input("Window Size", min_value=1, value=1, step=1)

PSOSL_particles = st.sidebar.number_input("Jumlah Partikel", min_value=1, value=18)

PSOSL_iters = st.sidebar.number_input("Jumlah Iterasi", min_value=1, value=5)

c1 = st.sidebar.number_input("c1", value=2.0)
c2 = st.sidebar.number_input("c2", value=2.0)
w = st.sidebar.number_input("w", value=0.7)

PSOSL_options = {'c1': c1, 'c2': c2, 'w': w}

st.sidebar.subheader("PSO Bounds")

units_min = st.sidebar.number_input("Units Min", value=16)
units_max = st.sidebar.number_input("Units Max", value=128)

lr_min = st.sidebar.number_input("Learning Rate Min", value=0.0001, format="%.4f")
lr_max = st.sidebar.number_input("Learning Rate Max", value=0.01, format="%.4f")

batch_min = st.sidebar.number_input("Batch Size Min", value=16)
batch_max = st.sidebar.number_input("Batch Size Max", value=128)

dropout_min = st.sidebar.number_input("Dropout Min", value=0.01, format="%.2f")
dropout_max = st.sidebar.number_input("Dropout Max", value=0.5, format="%.2f")

uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'])

# =====================================================
# LOAD DATA
# =====================================================
if uploaded_file is not None:

    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        emas = pd.read_csv(uploaded_file)
    else:
        emas = pd.read_excel(uploaded_file)

    # =====================================================
    # MISSING VALUE
    # =====================================================
    st.header("Missing Value")

    st.dataframe(emas.isnull().sum().reset_index())

    fig_missing = plt.figure(figsize=(10, 5))
    msno.matrix(emas)
    st.pyplot(fig_missing)

    # =====================================================
    # OUTLIER
    # =====================================================
    st.header("Outlier Detection")

    fig_box = plt.figure(figsize=(10, 5))
    sns.boxplot(x=emas['Terakhir'], color='gold')
    st.pyplot(fig_box)

    Q1 = emas['Terakhir'].quantile(0.25)
    Q3 = emas['Terakhir'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = emas[(emas['Terakhir'] < lower_bound) | (emas['Terakhir'] > upper_bound)]

    st.write("Outliers:", len(outliers))
    st.dataframe(outliers)

    # =====================================================
    # SPLIT
    # =====================================================
    st.header("Split Data")

    values = emas[['Terakhir']].values

    n = len(values)
    n_train = int(n * 0.8)

    train_values = values[:n_train]
    test_values = values[n_train:]

    st.write(n_train, n - n_train)

    # =====================================================
    # SCALING
    # =====================================================
    scaler_X = MinMaxScaler().fit(values[:n_train])
    scaler_y = MinMaxScaler().fit(values[:n_train])

    Xs = scaler_X.transform(values)
    ys = scaler_y.transform(values)

    # =====================================================
    # WINDOWING
    # =====================================================
    def make_sequences(X, y, window):
        Xs_seq, ys_seq = [], []
        for i in range(window, len(X)):
            Xs_seq.append(X[i-window:i])
            ys_seq.append(y[i])
        return np.array(Xs_seq), np.array(ys_seq)

    X_seq, y_seq = make_sequences(Xs, ys, window)

    split_idx = n_train - window

    X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
    X_test, y_test = X_seq[split_idx:], y_seq[split_idx:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # =====================================================
    # TRAIN BUTTON
    # =====================================================
    if st.button("Train GRU-PSO"):

        PSOSL_bounds = (
            [units_min, lr_min, batch_min, dropout_min],
            [units_max, lr_max, batch_max, dropout_max]
        )

        # ============================
        # CACHE CONFIG CHECK (REVISI)
        # ============================
        cache_path = "pso_cache/pso_reference.npz"

        reference_bounds = (
            [16, 0.0001, 16, 0.01],
            [128, 0.01, 128, 0.5]
        )

        reference_particles = 18
        reference_iters = 5
        reference_window = 1

        use_reference_cache = (
            list(PSOSL_bounds[0]) == list(reference_bounds[0])
            and list(PSOSL_bounds[1]) == list(reference_bounds[1])
            and PSOSL_particles == reference_particles
            and window == reference_window
        )

        # =====================================================
        # TRAIN-VAL SPLIT PSO
        # =====================================================
        val_ratio = 0.2
        n_tr = X_train.shape[0]
        n_val = int(n_tr * (1 - val_ratio))

        X_tr, y_tr = X_train[:n_val], y_train[:n_val]
        X_val, y_val = X_train[n_val:], y_train[n_val:]

        # =====================================================
        # PSO OBJECTIVE
        # =====================================================
        def make_obj(X_tr, y_tr, X_val, y_val, scaler_y):

            def obj(particles):

                costs = np.zeros(len(particles))

                for i, p in enumerate(particles):

                    units = int(np.round(p[0]))
                    lr = float(p[1])
                    batch = int(np.round(p[2]))
                    dropout = float(p[3])

                    try:
                        clear_session()

                        model = Sequential([
                            Input(shape=(X_tr.shape[1], 1)),
                            GRU(units),
                            Dropout(dropout),
                            Dense(1)
                        ])

                        model.compile(Adam(lr), "mse")

                        model.fit(X_tr, y_tr, epochs=10, batch_size=batch, verbose=0)

                        pred = model.predict(X_val, verbose=0)

                        pred = scaler_y.inverse_transform(pred)
                        true = scaler_y.inverse_transform(y_val.reshape(-1, 1))

                        costs[i] = mean_squared_error(true, pred)

                    except:
                        costs[i] = 1e12

                return costs

            return obj

        obj_fn = make_obj(X_tr, y_tr, X_val, y_val, scaler_y)

        optimizer = GlobalBestPSO(
            n_particles=PSOSL_particles,
            dimensions=4,
            options=PSOSL_options,
            bounds=PSOSL_bounds
        )

        # =====================================================
        # CACHE LOAD (REVISI)
        # =====================================================
        history_positions = []
        history_velocity = []
        history_costs = []
        history_gbest_cost = []
        history_gbest_pos = []

        iteration_results = []
        progress = st.progress(0)

        cached_positions = cached_velocity = None
        cached_costs = cached_gbest_cost = cached_gbest_pos = None

        if use_reference_cache and os.path.exists(cache_path):

            data = np.load(cache_path, allow_pickle=True)

            cached_positions = data["history_positions"]
            cached_velocity = data["history_velocity"]
            cached_costs = data["history_costs"]
            cached_gbest_cost = data["history_gbest_cost"]
            cached_gbest_pos = data["history_gbest_pos"]

            st.info("Cache digunakan (replay mode)")

        # =====================================================
        # PSO LOOP (REVISI UTAMA)
        # =====================================================
        max_iter = min(PSOSL_iters, len(cached_positions)) if cached_positions is not None else PSOSL_iters

        for it in range(max_iter):

            if cached_positions is not None:

                optimizer.swarm.position = cached_positions[it]
                optimizer.swarm.velocity = cached_velocity[it]

                costs = cached_costs[it]
                best_cost = cached_gbest_cost[it]
                best_pos = cached_gbest_pos[it]

                history_positions.append(cached_positions[it])
                history_velocity.append(cached_velocity[it])
                history_costs.append(costs)
                history_gbest_cost.append(float(best_cost))
                history_gbest_pos.append(best_pos)

            else:

                costs = obj_fn(optimizer.swarm.position)

                mask = costs < optimizer.swarm.pbest_cost

                optimizer.swarm.pbest_cost[mask] = costs[mask]
                optimizer.swarm.pbest_pos[mask] = optimizer.swarm.position[mask].copy()

                best = np.argmin(optimizer.swarm.pbest_cost)

                optimizer.swarm.best_cost = optimizer.swarm.pbest_cost[best]
                optimizer.swarm.best_pos = optimizer.swarm.pbest_pos[best].copy()

                history_positions.append(optimizer.swarm.position.copy())
                history_velocity.append(optimizer.swarm.velocity.copy())
                history_costs.append(costs)
                history_gbest_cost.append(float(optimizer.swarm.best_cost))
                history_gbest_pos.append(optimizer.swarm.best_pos.copy())

                r1 = np.random.rand(*optimizer.swarm.position.shape)
                r2 = np.random.rand(*optimizer.swarm.position.shape)

                optimizer.swarm.velocity = (
                    PSOSL_options['w'] * optimizer.swarm.velocity
                    + PSOSL_options['c1'] * r1 * (optimizer.swarm.pbest_pos - optimizer.swarm.position)
                    + PSOSL_options['c2'] * r2 * (optimizer.swarm.best_pos - optimizer.swarm.position)
                )

                optimizer.swarm.position += optimizer.swarm.velocity

                lb, ub = np.array(PSOSL_bounds[0]), np.array(PSOSL_bounds[1])
                optimizer.swarm.position = np.clip(optimizer.swarm.position, lb, ub)

            iteration_results.append({
                "Iterasi": it + 1,
                "Best Loss": history_gbest_cost[-1],
                "Units": int(np.round(history_gbest_pos[-1][0])),
                "Learning Rate": history_gbest_pos[-1][1],
                "Batch Size": int(np.round(history_gbest_pos[-1][2])),
                "Dropout": history_gbest_pos[-1][3]
            })

            progress.progress((it + 1) / max_iter)

        st.dataframe(pd.DataFrame(iteration_results))

        st.success("PSO selesai (cache-aware mode aktif)")

else:
    st.info("Silakan upload dataset terlebih dahulu.")
