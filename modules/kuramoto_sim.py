import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def run(coupling, n_oscillators):
    T = 10  # Teljes idő
    dt = 0.01  # Időlépés
    t = np.arange(0, T, dt)  # Itt definiáljuk a 't'-t ELŐSZÖR!

    theta = np.zeros((len(t), n_oscillators))
    omega = np.random.normal(1.0, 0.1, n_oscillators)
    theta[0] = np.random.uniform(0, 2*np.pi, n_oscillators)

    for i in range(1, len(t)):
        for j in range(n_oscillators):
            interaction = np.sum(np.sin(theta[i-1] - theta[i-1][j]))
            theta[i][j] = theta[i-1][j] + dt * (omega[j] + (coupling / n_oscillators) * interaction)

    fig, ax = plt.subplots()
    for i in range(n_oscillators):
        ax.plot(t, theta[:, i], label=f'Oszcillátor {i+1}')
    ax.set_title("Kuramoto szinkronizáció")
    ax.set_xlabel("Idő")
    ax.set_ylabel("Fázis")
    st.pyplot(fig)
