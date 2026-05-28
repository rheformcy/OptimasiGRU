import os
import gc
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
# SEED
# =====================================================
SEED_VALUE = 49
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

# =====================================================
# UI
# =====================================================
st.set_page_config(page_title="GRU-PSO Hybrid", layout="wide")
st.title("GRU-PSO Forecasting (Hybrid TRAIN + REPLAY)")

st.sidebar.header("Konfigurasi")

window = st.sidebar.number_input("Window", 1, value=1)

PSOSL_particles = st.sidebar.number_input("Particles", 1, value=18)
PSOSL_iters = st.sidebar.number_input("Iterations", 1, value=5)

c1 = st.sidebar.number_input("c1", value=2.0)
c2 = st.sidebar.number_input("c2", value=2.0)
w = st.sidebar.number_input("w", value=0.7)

PSOSL_options = {'c1': c1, 'c2': c2, 'w': w}

st.sidebar.subheader("Bounds")

units_min = st.sidebar.number_input("Units Min", value=16)
units_max = st.sidebar.number_input("Units Max", value=128)

lr_min = st.sidebar.number_input("LR Min", value=0.0001)
lr_max = st.sidebar.number_input("LR Max", value=0.01)

batch_min = st.sidebar.number_input("Batch Min", value=16)
batch_max = st.sidebar.number_input("Batch Max", value=128)

dropout_min = st.sidebar.number_input("Dropout Min", value=0.01)
dropout_max = st.sidebar.number_input("Dropout Max", value=0.5)

uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])

# =====================================================
# LOAD DATA
# =====================================================
if uploaded_file is not None:

    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    values = df[['Terakhir']].values

    n = len(values)
    n_train = int(n * 0.8)

    scaler = MinMaxScaler().fit(values[:n_train])
    scaled = scaler.transform(values)

    # windowing
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
    # SAFE COMPARISON (FIXED)
    # =====================================================
    
    def is_close(a, b, tol=1e-9):
        return abs(float(a) - float(b)) < tol
    
    bounds_current = [
        [units_min, lr_min, batch_min, dropout_min],
        [units_max, lr_max, batch_max, dropout_max]
    ]
    
    bounds_ref = reference_bounds
    
    use_cache = (
        os.path.exists(cache_path)
        and int(window) == int(reference_window)
        and int(PSOSL_particles) == int(reference_particles)
        and all(is_close(bounds_current[0][i], bounds_ref[0][i]) for i in range(4))
        and all(is_close(bounds_current[1][i], bounds_ref[1][i]) for i in range(4))
    )

        cached = None

        if use_cache:
            data = np.load(cache_path, allow_pickle=True)
            cached = {
                "pos": data["history_positions"],
                "vel": data["history_velocity"],
                "cost": data["history_costs"],
                "gbest_cost": data["history_gbest_cost"],
                "gbest_pos": data["history_gbest_pos"]
            }
            st.success("MODE: EXACT REPLAY (100% sama dengan Colab)")
        else:
            st.warning("MODE: TRAIN PSO (normal execution)")

        # =====================================================
        # VALIDATION SPLIT
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
                              epochs=10,
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
            n_particles=PSOSL_particles,
            dimensions=4,
            options=PSOSL_options,
            bounds=PSOSL_bounds
        )

        # =====================================================
        # LOOP (HYBRID CORE)
        # =====================================================
        results = []
        progress = st.progress(0)

        for it in range(PSOSL_iters):

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

            results.append({
                "Iterasi": it + 1,
                "Global Best": float(gbest_cost),
                "Best Iter": float(best_iter),
                "Units": int(np.round(gbest_pos[0])),
                "LR": float(gbest_pos[1]),
                "Batch": int(np.round(gbest_pos[2])),
                "Dropout": float(gbest_pos[3])
            })

            progress.progress((it + 1) / PSOSL_iters)

        st.dataframe(pd.DataFrame(results))

else:
    st.info("Upload dataset dulu")
