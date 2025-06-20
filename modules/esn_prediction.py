import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from time import time

def generate_lorenz(dt, steps, sigma=10.0, rho=28.0, beta=8.0/3.0):
    xyz = np.empty((steps + 1, 3))
    xyz[0] = (1.0, 1.0, 1.0)
    for i in range(steps):
        x, y, z = xyz[i]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        xyz[i + 1] = xyz[i] + dt * np.array([dx, dy, dz])
    return xyz

def embed_data(data, delay, dim, pred_steps):
    N = len(data)
    max_index = N - delay * (dim - 1) - pred_steps
    X, y = [], []
    for i in range(max_index):
        x_i = [data[i + j * delay] for j in range(dim)]
        y_i = data[i + delay * dim : i + delay * dim + pred_steps]
        if len(y_i) == pred_steps:
            X.append(x_i)
            y.append(y_i)
    return np.array(X), np.array(y)

def run():
    st.subheader("🧠 Echo State Network (ESN) predikció – Lorenz attractor")

    dt = st.slider("Időlépés (dt)", 0.01, 0.05, 0.03)
    steps = st.slider("Időlépések száma", 500, 3000, 1500)
    delay = st.slider("Késleltetés (delay)", 1, 20, 3)
    dim = st.slider("Beágyazás dimenziója", 2, 10, 5)
    pred_steps = st.slider("Előrejelzendő lépések", 1, 20, 1)

    st.markdown("⏳ Szimuláció fut...")

    start_time = time()
    data = generate_lorenz(dt, steps)
    progress = st.progress(0)

    X_data, y_data = [], []
    for i, idx in enumerate(range(3)):  # x, y, z komponensek
        X, y = embed_data(data[:, idx], delay, dim, pred_steps)
        X_data.append(X)
        y_data.append(y)
        progress.progress((i + 1) / 3.0)

    X_data = np.concatenate(X_data)
    y_data = np.concatenate(y_data)

    valid_mask = np.isfinite(X_data).all(axis=1) & np.isfinite(y_data).all(axis=1)
    X_data, y_data = X_data[valid_mask], y_data[valid_mask]

    model = Ridge(alpha=1.0)
    model.fit(X_data, y_data)

    y_pred = model.predict(X_data)
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))

    st.success(f"✅ RMSE: {rmse:.4f}")

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[:, 0], data[:, 1], data[:, 2], color='blue', label='Eredeti')
    ax.plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], color='orange', alpha=0.7, label='Predikció')
    ax.set_title("📊 Lorenz attractor – Predikció vs Eredeti")
    ax.legend()
    st.pyplot(fig)
