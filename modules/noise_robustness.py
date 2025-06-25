import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx

def simulate_kuramoto(N, K, noise_level, steps=500, dt=0.05):
    """Egyszerű Kuramoto-modell zajjal."""
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(1.0, 0.1, N)

    for _ in range(steps):
        interaction = np.sum(np.sin(np.subtract.outer(theta, theta)), axis=1)
        theta += (omega + (K / N) * interaction + np.random.normal(0, noise_level, N)) * dt

    return theta

def order_parameter(theta):
    """Rendparaméter számítása."""
    return np.abs(np.mean(np.exp(1j * theta)))

def run():
    st.title("📉 Zajtűrés és szinkronizációs robusztusság")

    N = st.slider("🎯 Oszcillátorok száma", 5, 100, 20)
    K = st.slider("🔗 Kapcsolódási erősség (K)", 0.0, 10.0, 2.0, step=0.1)
    max_noise = st.slider("🌀 Maximális zajszint", 0.0, 2.0, 1.0, step=0.05)
    steps = st.slider("⏱️ Iterációk száma", 100, 2000, 500, step=100)

    noise_levels = np.linspace(0, max_noise, 30)
    order_params = []

    st.write("🔁 Szimuláció zajlik...")

    for noise in noise_levels:
        theta = simulate_kuramoto(N, K, noise, steps=steps)
        r = order_parameter(theta)
        order_params.append(r)

    fig, ax = plt.subplots()
    ax.plot(noise_levels, order_params, marker="o")
    ax.set_xlabel("Zajszint")
    ax.set_ylabel("Rendparaméter (r)")
    ax.set_title("Szinkronizáció robusztussága zajjal szemben")
    ax.grid(True)
    st.pyplot(fig)
app = run
