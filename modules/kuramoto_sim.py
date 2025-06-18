import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🌐 Kuramoto szinkronizáció")
    st.write("Egyszerű Kuramoto-oszcillátorháló szimuláció.")

    n = 10
    coupling = 0.5
    timesteps = 100
    theta = np.random.rand(n) * 2 * np.pi
    omega = np.random.rand(n)
    history = [theta.copy()]
    sync_threshold = 0.99

    for _ in range(timesteps):
        for i in range(n):
            interaction = np.sum(np.sin(theta - theta[i]))
            theta[i] += omega[i] + (coupling / n) * interaction
        history.append(theta.copy())

        order_param = np.abs(np.sum(np.exp(1j * theta))) / n
        if order_param > sync_threshold:
            break

    data = np.array(history)
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title("Kuramoto szinkronizáció")
    ax.set_xlabel("Iteráció")
    ax.set_ylabel("Fázis")
    st.pyplot(fig)
    st.success(f"Szinkronizáció elérve {len(data)} iteráció alatt.")
