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
    st.subheader("📊 Szinkronizációs robusztusság (Gyorsított Pro változat)")

    N = st.slider("Oszcillátorok száma (N)", 10, 100, 30)
    T = st.slider("Iterációk száma (T)", 100, 1000, 250)
    dt = st.slider("Időlépés (dt)", 0.01, 0.1, 0.03)

    k_min, k_max = st.slider("Kapcsolási erősség tartománya (K)", 0.0, 10.0, (0.5, 3.0))
    noise_min, noise_max = st.slider("Zaj tartomány (szórás)", 0.0, 2.0, (0.1, 0.5))

    k_values = np.linspace(k_min, k_max, 10)
    noise_values = np.linspace(noise_min, noise_max, 10)

    sync_matrix = np.zeros((len(noise_values), len(k_values)))

    progress = st.empty()
    for i, noise in enumerate(noise_values):
        for j, K in enumerate(k_values):
            theta = np.random.uniform(0, 2 * np.pi, N)
            omega = np.random.normal(0, 1, N)
            for _ in range(T):
                theta = kuramoto_step(theta, omega, K, noise, dt)
            sync_matrix[i, j] = compute_sync_index(theta)
        progress.progress((i + 1) / len(noise_values))

    fig, ax = plt.subplots()
    c = ax.imshow(sync_matrix, aspect='auto', origin='lower',
                  extent=[k_min, k_max, noise_min, noise_max],
                  cmap='plasma')
    fig.colorbar(c, ax=ax, label='Szinkronizációs index')
    ax.set_title("🧪 Robusztussági térkép (Gyorsított)")
    ax.set_xlabel("Kapcsolási erősség (K)")
    ax.set_ylabel("Zaj (szórás)")

    st.pyplot(fig)
