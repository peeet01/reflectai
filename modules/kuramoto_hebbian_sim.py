import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def run():
    st.subheader("🔁 Kuramoto–Hebbian háló szimuláció")

    # Paraméterek beállítása
    N = st.slider("Neuronok / oszcillátorok száma", 5, 50, 15)
    K = st.slider("Kuramoto kapcsolási erősség", 0.0, 10.0, 2.0)
    eta = st.slider("Hebbian tanulási ráta", 0.0, 1.0, 0.05)
    T = st.slider("Szimuláció lépésszáma", 10, 500, 200)

    # Kezdeti frekvenciák és fázisok
    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)
    initial_theta = theta.copy()

    # Kezdeti szinaptikus súlymátrix (Hebbian tanulás)
    W = np.random.uniform(0.0, 1.0, (N, N))
    np.fill_diagonal(W, 0)  # nincs önkapcsolat
    W = (W + W.T) / 2  # szimmetrikus

    dt = 0.05
    sync_history = []
    weight_history = []

    # Szimuláció
    for t in range(T):
        theta_matrix = np.subtract.outer(theta, theta)
        coupling = np.sum(W * np.sin(theta_matrix), axis=1)
        theta += (omega + (K / N) * coupling) * dt

        # Hebbian tanulás - frissítés
        phase_diff = np.subtract.outer(theta, theta)
        W += eta * np.cos(phase_diff) * dt
        np.fill_diagonal(W, 0)
        W = np.clip(W, 0.0, 1.0)  # súlykorlát

        # Szinkronizációs index mentése
        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        sync_history.append(r)
        if t % 10 == 0:
            weight_history.append(W.copy())

    # Ábrák
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"polar": [True, True]})
    axs[0].set_title("Kezdeti fáziseloszlás")
    axs[0].scatter(initial_theta, np.ones(N), color='blue', alpha=0.7)

    axs[1].set_title(f"Végső fáziseloszlás\nSzinkronizációs index: r = {sync_history[-1]:.2f}")
    axs[1].scatter(theta, np.ones(N), color='red', alpha=0.7)
    st.pyplot(fig)

    # Szinkronizációs index alakulása
    st.line_chart(sync_history, height=200)

    # Súlymátrix vizualizáció (utolsó állapot)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    im = ax2.imshow(W, cmap='viridis')
    plt.colorbar(im, ax=ax2)
    ax2.set_title("🧠 Tanult Hebbian súlymátrix (utolsó állapot)")
    st.pyplot(fig2)

    st.info("Ez a modell ötvözi a Kuramoto-szinkronizációt és a Hebbian tanulást dinamikusan változó kapcsolatokkal.")
