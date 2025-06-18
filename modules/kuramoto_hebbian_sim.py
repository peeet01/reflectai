import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🔁 Kuramoto–Hebbian háló")

    st.write("Adaptív Kuramoto szinkronizáció Hebbian tanulással.")

    N = 10
    steps = 200
    dt = 0.1
    K = 0.5
    learning_rate = 0.01

    theta = np.random.rand(N) * 2 * np.pi
    W = np.random.rand(N, N)
    np.fill_diagonal(W, 0)

    theta_hist = [theta.copy()]
    coherence_hist = []

    for step in range(steps):
        theta_dot = np.zeros(N)
        for i in range(N):
            interaction = np.sum(W[i, j] * np.sin(theta[j] - theta[i]) for j in range(N))
            theta_dot[i] = K * interaction
        theta += dt * theta_dot
        theta_hist.append(theta.copy())

        # Hebbian frissítés
        for i in range(N):
            for j in range(N):
                if i != j:
                    W[i, j] += learning_rate * np.cos(theta[i] - theta[j])

        # Koherencia mérése (r)
        r = np.abs(np.sum(np.exp(1j * theta))) / N
        coherence_hist.append(r)

    # Vizualizáció
    theta_hist = np.array(theta_hist)
    fig, ax = plt.subplots()
    for i in range(N):
        ax.plot(theta_hist[:, i], label=f"Oszcillátor {i}")
    ax.set_title("Fázisok időben (adaptív)")
    ax.set_xlabel("Iteráció")
    ax.set_ylabel("Fázis")
    st.pyplot(fig)

    # Koherencia
    fig2, ax2 = plt.subplots()
    ax2.plot(coherence_hist)
    ax2.set_title("Koherencia érték alakulása")
    ax2.set_xlabel("Iteráció")
    ax2.set_ylabel("r")
    st.pyplot(fig2)

    st.success("Kuramoto–Hebbian háló szimuláció lefutott.")
