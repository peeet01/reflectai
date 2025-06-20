import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz_system(state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def run():
    st.subheader("🌪️ Lorenz attraktor – Kaotikus dinamika vizualizáció")

    # Paraméterek beállítása
    sigma = st.slider("σ (Prandtl-szám)", 0.0, 20.0, 10.0)
    rho = st.slider("ρ (Rayleigh-szám)", 0.0, 50.0, 28.0)
    beta = st.slider("β", 0.0, 10.0, 8.0 / 3.0)
    T = st.slider("Szimuláció ideje (lépések)", 100, 10000, 2000, step=100)
    dt = 0.01
    color_scheme = st.selectbox("🎨 Színpaletta", ["viridis", "plasma", "cividis", "magma", "cool"])

    # Kezdeti állapot
    state = np.array([1.0, 1.0, 1.0])
    trajectory = np.zeros((T, 3))

    for t in range(T):
        trajectory[t] = state
        state = state + lorenz_system(state, sigma, rho, beta) * dt

    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    # 3D Lorenz attraktor
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(x, y, z, lw=0.5, color='black')
    ax.set_title("🌀 Lorenz attraktor")

    # Idősor plot
    ax2 = fig.add_subplot(122)
    ax2.plot(np.arange(T) * dt, x, label='x(t)', alpha=0.8)
    ax2.plot(np.arange(T) * dt, y, label='y(t)', alpha=0.8)
    ax2.plot(np.arange(T) * dt, z, label='z(t)', alpha=0.8)
    ax2.set_title("📈 Idősorok")
    ax2.set_xlabel("t")
    ax2.legend()

    st.pyplot(fig)
