import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run():
    st.subheader("🎛️ Generatív Kuramoto szimuláció – Fázisfejlődés és Szinkronizáció")

    # Paraméterek
    N = st.slider("Oszcillátorok száma (N)", 5, 100, 20)
    K = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    T = st.slider("Szimulációs idő (lépések)", 10, 500, 200)

    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2*np.pi, N)
    initial_theta = theta.copy()

    dt = 0.05
    r_values = []

    for _ in range(T):
        theta_matrix = np.subtract.outer(theta, theta)
        coupling = np.sum(np.sin(theta_matrix), axis=1)
        theta += (omega + (K / N) * coupling) * dt
        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        r_values.append(r)

    final_theta = theta.copy()

    # Ábrák
    fig1, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw=dict(polar=True))
    axes[0].set_title("🌀 Kezdeti fáziseloszlás")
    axes[0].scatter(initial_theta, np.ones(N), c='blue', alpha=0.75)
    axes[1].set_title(f"🔄 Végső fáziseloszlás\nSzinkronizációs index: r = {r_values[-1]:.2f}")
    axes[1].scatter(final_theta, np.ones(N), c='red', alpha=0.75)
    st.pyplot(fig1)

    # Szinkronizáció időbeli változása
    fig2, ax2 = plt.subplots()
    ax2.plot(r_values, color='green')
    ax2.set_title("📈 Szinkronizációs index időben")
    ax2.set_xlabel("Lépések")
    ax2.set_ylabel("r érték")
    ax2.grid(True)
    st.pyplot(fig2)
