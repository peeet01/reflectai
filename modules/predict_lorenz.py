import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D


def lorenz_system(x, y, z, s=10, r=28, b=2.667):
    dx = s * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z
    return dx, dy, dz


def generate_lorenz_data(dt, num_steps):
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)

    xs[0], ys[0], zs[0] = (0., 1., 1.05)

    for i in range(num_steps):
        dx, dy, dz = lorenz_system(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (dx * dt)
        ys[i + 1] = ys[i] + (dy * dt)
        zs[i + 1] = zs[i] + (dz * dt)

    return xs, ys, zs


def embed_data(data, delay, dimension, pred_horizon=1):
    X = []
    y = []
    for i in range(len(data) - delay * (dimension - 1) - pred_horizon):
        X.append([data[i + j * delay] for j in range(dimension)])
        y.append(data[i + delay * (dimension - 1) + pred_horizon])
    return np.array(X), np.array(y)


def run():
    st.subheader("📈 Lorenz attractor predikció")

    dt = st.slider("Időlépés (dt)", 0.001, 0.05, 0.03)
    num_steps = st.slider("Időlépések száma", 100, 2000, 1500)
    delay = st.slider("Késleltetés (delay)", 1, 20, 3)
    dimension = st.slider("Beágyazás dimenziója", 2, 10, 5)
    pred_horizon = st.slider("Előrejelzendő lépések", 1, 50, 1)

    with st.spinner("🔄 Szimuláció..."):
        xs, ys, zs = generate_lorenz_data(dt, num_steps)

        # Embed minden dimenzióra
        X_x, y_x = embed_data(xs, delay, dimension, pred_horizon)
        X_y, y_y = embed_data(ys, delay, dimension, pred_horizon)
        X_z, y_z = embed_data(zs, delay, dimension, pred_horizon)

        # Ellenőrzés: az adathosszak egyezzenek
        min_len = min(len(X_x), len(X_y), len(X_z))
        X_data = np.hstack((X_x[:min_len], X_y[:min_len], X_z[:min_len]))
        y_data = np.vstack((y_x[:min_len], y_y[:min_len], y_z[:min_len])).T

        # Hibaellenőrzés: csak véges értékek
        valid_mask = np.all(np.isfinite(X_data), axis=1) & np.all(np.isfinite(y_data), axis=1)
        X_data = X_data[valid_mask]
        y_data = y_data[valid_mask]

        if len(X_data) == 0 or len(y_data) == 0:
            st.error("❌ Érvénytelen adatok – próbálj kisebb késleltetést vagy dimenziót.")
            return

        # Modell tanítása
        model = Ridge(alpha=1.0)
        model.fit(X_data, y_data)
        y_pred = model.predict(X_data)

        rmse = np.sqrt(mean_squared_error(y_data, y_pred))
        st.success(f"📉 RMSE hiba: {rmse:.4f}")

        # Megjelenítés
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(xs, ys, zs, lw=0.5, color='blue', label='Eredeti')
        ax1.set_title("🎯 Eredeti Lorenz attractor")

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], lw=0.5, color='red', label='Predikció')
        ax2.set_title("🔮 Predikált pálya")

        st.pyplot(fig)
