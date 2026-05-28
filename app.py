import os
import random
import warnings

warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session

from pyswarms.single.global_best import GlobalBestPSO

# =====================================================
# SEED (REPRODUCIBLE)
# =====================================================
SEED = 49
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================
# UI
# =====================================================
st.set_page_config(page_title="GRU-PSO Hybrid", layout="wide")
st.title("GRU-PSO Forecasting (TRAIN + EXACT REPLAY)")

st.sidebar.header("Config")

window = st.sidebar.number_input("Window", min_value=1, value=1)

particles = st.sidebar.number_input("Particles", min_value=1, value=18)
iters = st.sidebar.number_input("Iterations", min_value=1, value=5)

c1 = st.sidebar.number_input("c1", value=2.0)
c2 = st.sidebar.number_input("c2", value=2.0)
w = st.sidebar.number_input("w", value=0.7)

bounds = (
    [
        st.sidebar.number_input("Units Min", value=16),
        st.sidebar.number_input("LR Min", value=0.0001),
        st.sidebar.number_input("Batch Min", value=16),
        st.sidebar.number_input("Dropout Min", value=0.01),
    ],
    [
        st.sidebar.number_input("Units Max", value=128),
        st.sidebar.number_input("LR Max", value=0.01),
        st.sidebar.number_input("Batch Max", value=128),
        st.sidebar.number_input("Dropout Max", value=0.5),
    ],
)

uploaded = st.file_uploader("Upload CSV/XLSX")

# =====================================================
# LOAD DATA
# =====================================================
if uploaded is not None:

    if uploaded.name.endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    values = df[['Terakhir']].values

    n = len(values)
    n_train = int(n * 0.8)

    scaler = MinMaxScaler().fit(values[:n_train])
    scaled = scaler.transform(values)

    # =====================================================
    # WINDOWING
    # =====================================================
    def make_seq(X, w):
        Xs, ys = [], []
        for i in range(w, len(X)):
            Xs.append(X[i-w:i])
            ys.append(X[i])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = make_seq(scaled, window)

    split = n_train - window
    X_train, y_train = X_seq[:split], y_seq[:split]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # =====================================================
    # CACHE CONFIG
    # =====================================================
    cache_path = "pso_cache/pso_reference.npz"

    reference_particles = 18
    reference_window = 1
    reference_bounds = (
        [16, 0.0001, 16, 0.01],
        [128, 0.01, 128, 0.5]
    )

    def close(a, b):
        return abs(float(a) - float(b)) < 1e-9

    use_cache = (
        os.path.exists(cache_path)
        and int(window) == reference_window
        and int(particles) == reference_particles
        and all(close(bounds[0][i], reference_bounds[0][i]) for i in range(4))
        and all(close(bounds[1][i], reference_bounds[1][i]) for i in range(4))
    )

    cached = None

    if use_cache:
        data = np.load(cache_path, allow_pickle=True)

        cached = {
            "pos": data["history_positions"],
            "cost": data["history_costs"],
            "gbest_cost": data["history_gbest_cost"],
            "gbest_pos": data["history_gbest_pos"],
        }

        st.success("MODE: EXACT REPLAY (100% dari Colab)")
    else:
        st.warning("MODE: TRAIN PSO")

    # =====================================================
    # SPLIT TRAIN/VAL
    # =====================================================
    n_tr = X_train.shape[0]
    n_val = int(n_tr * 0.8)

    X_tr, y_tr = X_train[:n_val], y_train[:n_val]
    X_val, y_val = X_train[n_val:], y_train[n_val:]

    # =====================================================
    # OBJECTIVE FUNCTION
    # =====================================================
    def obj_fn(particles):

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

                model.fit(X_tr, y_tr,
                          epochs=5,
                          batch_size=batch,
                          verbose=0)

                pred = model.predict(X_val, verbose=0)

                pred = scaler.inverse_transform(pred)
                true = scaler.inverse_transform(y_val.reshape(-1, 1))

                costs[i] = mean_squared_error(true, pred)

            except:
                costs[i] = 1e12

        return costs

    optimizer = GlobalBestPSO(
        n_particles=particles,
        dimensions=4,
        options={'c1': c1, 'c2': c2, 'w': w},
        bounds=bounds
    )

    # =====================================================
    # LOOP CORE (ANTI ERROR)
    # =====================================================
    results = []
    progress = st.progress(0)

    max_iter = iters

    for it in range(max_iter):

        if use_cache:

            pos = cached["pos"][it]
            cost = cached["cost"][it]
            gbest_cost = cached["gbest_cost"][it]
            gbest_pos = cached["gbest_pos"][it]

            best_iter = np.min(cost)

        else:

            cost = obj_fn(optimizer.swarm.position)

            mask = cost < optimizer.swarm.pbest_cost

            optimizer.swarm.pbest_cost[mask] = cost[mask]
            optimizer.swarm.pbest_pos[mask] = optimizer.swarm.position[mask].copy()

            best = np.argmin(optimizer.swarm.pbest_cost)

            optimizer.swarm.best_cost = optimizer.swarm.pbest_cost[best]
            optimizer.swarm.best_pos = optimizer.swarm.pbest_pos[best].copy()

            gbest_cost = optimizer.swarm.best_cost
            gbest_pos = optimizer.swarm.best_pos
            best_iter = np.min(cost)

        results.append({
            "Iterasi": it + 1,
            "Global Best": float(gbest_cost),
            "Best Iter": float(best_iter),
            "Units": int(np.round(gbest_pos[0])),
            "LR": float(gbest_pos[1]),
            "Batch": int(np.round(gbest_pos[2])),
            "Dropout": float(gbest_pos[3])
        })

        progress.progress((it + 1) / max_iter)

    st.dataframe(pd.DataFrame(results))

else:
    st.info("Upload dataset dulu")
