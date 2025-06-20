import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error

def lorenz(x, y, z, s=10, r=28, b=2.667):
    dx = s * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return dx, dy, dz

def simulate_lorenz(x0, y0, z0, dt=0.01, steps=1000):
    xs = np.empty(steps)
    ys = np.empty(steps)
    zs = np.empty(steps)
    xs[0], ys[0], zs[0] = x0, y0, z0
    for i in range(1, steps):
        dx, dy, dz = lorenz(xs[i-1], ys[i-1], zs[i-1])
        xs[i] = xs[i-1] + dx * dt
        ys[i] = ys[i-1] + dy * dt
        zs[i] = zs[i-1] + dz * dt
    return xs, ys, zs

def simple_predictor(xs, ys, zs):
    # Triviális késleltetéses modell előrejelzéshez
    return xs[:-1], ys[:-1], zs[:-1]

def run():
    st.subheader("📈 Lorenz attractor predikció")

    dt = st.slider("Időlépés (dt)", 0.001, 0.05, 0.01)
    steps = st.slider("Időlépések száma", 100, 2000, 1000)

    st.markdown("🔄 Szimuláció...")
    x0, y0, z0 = 0., 1., 1.05
    xs, ys, zs = simulate_lorenz(x0, y0, z0, dt, steps)

    pred_x, pred_y, pred_z = simple_predictor(xs, ys, zs)
    target_x, target_y, target_z = xs[1:], ys[1:], zs[1:]

    error = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2 + (pred_z - target_z)**2)
    rmse = np.sqrt(mean_squared_error(np.vstack([target_x, target_y, target_z]).T,
                                      np.vstack([pred_x, pred_y, pred_z]).T))

    st.markdown(f"🔍 **RMSE**: {rmse:.4f}")

    fig = plt.figure(figsize=(14, 4))

    # 3D attraktor
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(xs, ys, zs, color='blue', label="Valódi")
    ax1.plot(pred_x, pred_y, pred_z, color='red', alpha=0.6, label="Predikció")
    ax1.set_title("🌪️ Lorenz attraktor")
    ax1.legend()

    # Hiba időfüggése
    ax2 = fig.add_subplot(132)
    ax2.plot(error, color='purple')
    ax2.set_title("📉 Hiba időfüggése")
    ax2.set_xlabel("Időlépés")
    ax2.set_ylabel("Hiba")

    # Hiba eloszlása
    ax3 = fig.add_subplot(133)
    ax3.hist(error, bins=30, color='green', alpha=0.7)
    ax3.set_title("📊 Hiba eloszlása")
    ax3.set_xlabel("Hiba")
    ax3.set_ylabel("Gyakoriság")

    st.pyplot(fig)
