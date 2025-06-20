import streamlit as st
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def lorenz_system(state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def run():
    st.subheader("🌀 Lorenz-szimuláció (3D Kaotikus rendszer)")

    # Paraméterek beállítása
    sigma = st.slider("📐 Sigma", 0.0, 30.0, 10.0)
    rho = st.slider("📏 Rho", 0.0, 50.0, 28.0)
    beta = st.slider("📉 Beta", 0.0, 10.0, 2.67)

    steps = st.slider("⏱️ Iterációk száma", 100, 10000, 5000)
    dt = st.slider("🕒 Időlépés (dt)", 0.001, 0.01, 0.005)

    # Kezdeti állapot
    state = np.array([0.1, 0.0, 0.0])
    trajectory = [state]

    for _ in range(steps):
        deriv = lorenz_system(state, sigma, rho, beta)
        state = state + deriv * dt
        trajectory.append(state)

    trajectory = np.array(trajectory)

    # Ábra megjelenítés
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=0.5, color='purple')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='red', s=30)
    ax.set_title("🌪️ Lorenz Attractor (Kaotikus pálya)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    st.pyplot(fig)
