# modules/kuramoto_hebbian_sim.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def simulate_kuramoto_hebbian(N=10, K=1.0, eta=0.01, T=10, dt=0.1):
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(1.0, 0.1, N)
    W = np.ones((N, N)) - np.eye(N)

    time_points = int(T / dt)
    sync = []

    for t in range(time_points):
        theta_diff = theta[:, None] - theta[None, :]
        dtheta = omega + (K / N) * np.sum(W * np.sin(-theta_diff), axis=1)
        theta += dtheta * dt
        W += eta * np.cos(theta_diff) * dt
        np.fill_diagonal(W, 0)

        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        sync.append(r)

    return sync

def run():
    st.header("🧠 Kuramoto–Hebbian háló")

    N = st.slider("Oszcillátorok száma", 5, 50, 10)
    K = st.slider("Kapcsolási erősség", 0.0, 5.0, 1.0)
    eta = st.slider("Tanulási ráta (Hebbian)", 0.001, 0.1, 0.01)
    T = st.slider("Szimuláció ideje", 5, 50, 10)

    sync = simulate_kuramoto_hebbian(N=N, K=K, eta=eta, T=T)

    fig, ax = plt.subplots()
    ax.plot(sync)
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("Szinkronizáció mértéke")
    ax.set_title("Kuramoto–Hebbian szinkronizáció")
    st.pyplot(fig)
app = run
