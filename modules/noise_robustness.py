import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def kuramoto_sim(N, T, dt, K, noise_std):
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)
    order_params = []

    for _ in range(T):
        theta_matrix = np.subtract.outer(theta, theta)
        coupling = np.sum(np.sin(theta_matrix), axis=1)
        noise = np.random.normal(0, noise_std, N)
        theta += (omega + (K / N) * coupling + noise) * dt
        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        order_params.append(r)

    return np.mean(order_params)

def run():
    st.subheader("🔊 Zajtűrés és szinkronizáció robusztusság")

    N = st.slider("🧠 Oszcillátorok száma", 5, 100, 20)
    T = st.slider("⏱️ Iterációk száma", 50, 500, 200)
    dt = st.slider("🕒 Időlépés", 0.01, 0.1, 0.03)
    num_K = st.slider("📈 K (kapcsolódás) felbontás", 5, 20, 10)
    num_noise = st.slider("📉 Zajszint felbontás", 5, 20, 10)

    K_vals = np.linspace(0.0, 10.0, num_K)
    noise_vals = np.linspace(0.0, 2.0, num_noise)
    R_matrix = np.zeros((num_K, num_noise))

    progress = st.progress(0.0, text="Szimuláció fut...")

    for i, K in enumerate(K_vals):
        for j, noise in enumerate(noise_vals):
            try:
                r_mean = kuramoto_sim(N, T, dt, K, noise)
                R_matrix[i, j] = r_mean
            except Exception as e:
                R_matrix[i, j] = 0
        progress.progress((i + 1) / num_K, text=f"{int((i + 1) / num_K * 100)}% kész")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(R_matrix, origin='lower', aspect='auto',
                   extent=[noise_vals[0], noise_vals[-1], K_vals[0], K_vals[-1]],
                   cmap='viridis')

    ax.set_xlabel("Zaj szórása (σ)")
    ax.set_ylabel("Kapcsolási erősség (K)")
    ax.set_title("🌀 Átlagos szinkronizációs index (r)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Átlagos r")

    st.pyplot(fig)
