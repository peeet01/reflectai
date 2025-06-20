import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def lorenz(x, y, z, s=10, r=28, b=2.667):
    dx = s*(y - x)
    dy = x*(r - z) - y
    dz = x*y - b*z
    return dx, dy, dz

def simple_predictor(xs, ys, zs):
    return xs[:-1], ys[:-1], zs[:-1]

def run():
    st.subheader("📈 Lorenz attractor predikció")

    dt = st.slider("Időlépés (dt)", 0.001, 0.05, 0.01)
    steps = st.slider("Időlépések száma", 100, 2000, 1000)

    xs = np.empty(steps + 1)
    ys = np.empty(steps + 1)
    zs = np.empty(steps + 1)

    xs[0], ys[0], zs[0] = (0., 1., 1.05)

    for i in range(steps):
        dx, dy, dz = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + dx * dt
        ys[i + 1] = ys[i] + dy * dt
        zs[i + 1] = zs[i] + dz * dt

    pred_x, pred_y, pred_z = simple_predictor(xs, ys, zs)
    target_x, target_y, target_z = xs[1:], ys[1:], zs[1:]

    # Hossz egyeztetés
    min_len = min(len(pred_x), len(target_x))
    pred_x, pred_y, pred_z = pred_x[:min_len], pred_y[:min_len], pred_z[:min_len]
    target_x, target_y, target_z = target_x[:min_len], target_y[:min_len], target_z[:min_len]

    error = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2 + (pred_z - target_z)**2)
    rmse = np.sqrt(mean_squared_error(
        np.vstack([target_x, target_y, target_z]).T,
        np.vstack([pred_x, pred_y, pred_z]).T
    ))

    st.markdown(f"📉 **RMSE (gyökös átlagos négyzetes hiba): {rmse:.4f}**")

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    
    axs[0].plot(xs, ys, zs, lw=0.5)
    axs[0].set_title("🌪️ Lorenz attractor pálya")
    axs[0].set_xlabel("X"); axs[0].set_ylabel("Y")

    axs[1].plot(target_x, label="Valós")
    axs[1].plot(pred_x, '--', label="Predikció")
    axs[1].set_title("🔍 X predikció vs. valós")
    axs[1].legend()

    axs[2].plot(error, color='red')
    axs[2].set_title("📊 Predikciós hiba alakulása")
    axs[2].set_xlabel("Lépés"); axs[2].set_ylabel("Hiba")

    st.pyplot(fig)
