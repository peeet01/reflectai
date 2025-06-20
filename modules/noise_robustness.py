import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def kuramoto_step(theta, omega, K, noise_std, dt):
    N = len(theta)
    theta_matrix = np.subtract.outer(theta, theta)
    coupling = np.sum(np.sin(theta_matrix), axis=1)
    noise = np.random.normal(0, noise_std, N)
    return theta + (omega + (K / N) * coupling + noise) * dt

def compute_sync_index(theta):
    return np.abs(np.mean(np.exp(1j * theta)))

def run():
    st.subheader("📊 Szinkronizációs robusztusság (Pro)")

    N = st.slider("Oszcillátorok száma (N)", 10, 100, 30)
    T = st.slider("Iterációk száma (T)", 100, 1000, 300)
    dt = st.slider("Időlépés (dt)", 0.01, 0.1, 0.05)

    k_min, k_max = st.slider("Kapcsolási erősség tartománya (K)", 0.0, 10.0, (1.0, 5.0))
    noise_min, noise_max = st.slider("Zaj tartomány (szórás)", 0.0, 2.0, (0.0, 1.0))

    k_values = np.linspace(k_min, k_max, 20)
    noise_values = np.linspace(noise_min, noise_max, 20)

    sync_matrix = np.zeros((len(noise_values), len(k_values)))

    progress = st.progress(0.0, "Szimuláció fut...")

    for i, noise in enumerate(noise_values):
        for j, K in enumerate(k_values):
            theta = np.random.uniform(0, 2 * np.pi, N)
            omega = np.random.normal(0, 1, N)
            for _ in range(T):
                theta = kuramoto_step(theta, omega, K, noise, dt)
            sync_index = compute_sync_index(theta)
            sync_matrix[i, j] = sync_index
        progress.progress((i + 1) / len(noise_values))

    fig, ax = plt.subplots()
    c = ax.imshow(sync_matrix, aspect='auto', origin='lower',
                  extent=[k_min, k_max, noise_min, noise_max],
                  cmap='viridis')
    fig.colorbar(c, ax=ax, label='Szinkronizációs index')
    ax.set_title("🧪 Szinkronizációs robusztusság térképe")
    ax.set_xlabel("Kapcsolási erősség (K)")
    ax.set_ylabel("Zaj szórás")

    st.pyplot(fig)
