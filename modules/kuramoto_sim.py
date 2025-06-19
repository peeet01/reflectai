import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run(n_oscillators=10, coupling_strength=1.0, sim_time=10):
    st.subheader("🌀 Kuramoto szinkronizáció")

    dt = 0.05
    t = np.arange(0, sim_time, dt)

    theta = np.zeros((len(t), n_oscillators))
    theta[0] = np.random.uniform(0, 2 * np.pi, n_oscillators)
    omega = np.random.normal(0.0, 1.0, n_oscillators)

    for i in range(1, len(t)):
        for j in range(n_oscillators):
            interaction = np.sum(np.sin(theta[i - 1] - theta[i - 1, j]))
            theta[i, j] = theta[i - 1, j] + dt * (
                omega[j] + (coupling_strength / n_oscillators) * interaction
            )

    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(n_oscillators):
        ax.plot(t, theta[:, i], label=f'Oszcillátor {i+1}')
    ax.set_title("Kuramoto szinkronizáció fázisidősora")
    ax.set_xlabel("Idő (s)")
    ax.set_ylabel("Fázis (rad)")
    st.pyplot(fig)

    # Szinkronizációs mérték
    r_values = np.abs(np.mean(np.exp(1j * theta), axis=1))
    st.line_chart(r_values, height=200, use_container_width=True)
    st.caption("🔁 Szinkronizáció mértéke az idő függvényében (0–1 között)")
