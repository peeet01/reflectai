# modules/lorenz_sim.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz_system(x, y, z, sigma, rho, beta):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def run():
    st.header("🌀 Lorenz szimuláció")
    sigma = st.slider("σ (Prandtl-szám)", 0.0, 20.0, 10.0)
    rho = st.slider("ρ (Rayleigh-szám)", 0.0, 50.0, 28.0)
    beta = st.slider("β", 0.0, 10.0, 8.0 / 3.0)

    dt = 0.01
    steps = 10000

    x = np.empty(steps)
    y = np.empty(steps)
    z = np.empty(steps)
    x[0], y[0], z[0] = (0., 1., 1.05)

    for i in range(1, steps):
        dx, dy, dz = lorenz_system(x[i - 1], y[i - 1], z[i - 1], sigma, rho, beta)
        x[i] = x[i - 1] + dx * dt
        y[i] = y[i - 1] + dy * dt
        z[i] = z[i - 1] + dz * dt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, lw=0.5)
    ax.set_title("Lorenz attraktor")
    st.pyplot(fig)
app = run
