import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run():
    st.subheader("🎛️ Generatív Kuramoto szimuláció – Kezdeti vs. Végső állapot")

    # Paraméterek
    N = st.slider("Oszcillátorok száma (N)", 5, 100, 20)
    K = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    T = st.slider("Szimulációs idő (lépések)", 10, 500, 200)

    # Frekvenciák és kezdeti fázisok
    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2*np.pi, N)
    initial_theta = theta.copy()  # Mentés

    dt = 0.05

    # Kuramoto egyenlet iterációja
    for _ in range(T):
        theta_matrix = np.subtract.outer(theta, theta)
        coupling = np.sum(np.sin(theta_matrix), axis=1)
        theta += (omega + (K / N) * coupling) * dt

    # Szinkronizációs index számítása
    order_parameter = np.abs(np.sum(np.exp(1j * theta)) / N)

    # Ábrák
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw=dict(polar=True))
    axes[0].set_title("🌀 Kezdeti fáziseloszlás")
    axes[0].scatter(initial_theta, np.ones(N), c='blue', alpha=0.75)

    axes[1].set_title(f"🔄 Végső fáziseloszlás\nSzinkronizációs index: r = {order_parameter:.2f}")
    axes[1].scatter(theta, np.ones(N), c='red', alpha=0.75)

    st.pyplot(fig)
