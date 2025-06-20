import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def kuramoto_sim(N, T, dt, K, noise_std):
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)

    for _ in range(T):
        theta_matrix = np.subtract.outer(theta, theta)
        coupling = np.sum(np.sin(theta_matrix), axis=1)
        noise = np.random.normal(0, noise_std, N)
        theta += (omega + (K / N) * coupling + noise) * dt

    r = np.abs(np.sum(np.exp(1j * theta)) / N)
    return r

def run():
    st.subheader("🔉 Szinkronizáció robusztussága zaj esetén")

    N = st.slider("🧩 Oszcillátorok száma", 10, 100, 30)
    T = st.slider("⏱️ Iterációk száma", 100, 1000, 300)
    dt = st.slider("🧮 Időlépés", 0.01, 0.1, 0.03)
    num_K = st.slider("📊 K felbontás", 5, 30, 15)
    num_noise = st.slider("📊 Zaj felbontás", 5, 30, 15)

    K_vals = np.linspace(0, 5, num_K)
    noise_vals = np.linspace(0, 2, num_noise)
    R_matrix = np.zeros((num_noise, num_K))

    progress = st.progress(0.0)
    total = num_K * num_noise

    for i, noise_std in enumerate(noise_vals):
        for j, K in enumerate(K_vals):
            R_matrix[i, j] = kuramoto_sim(N, T, dt, K, noise_std)
            progress.progress(((i * num_K + j + 1) / total))

    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.imshow(R_matrix, extent=[K_vals[0], K_vals[-1], noise_vals[0], noise_vals[-1]],
                  origin='lower', aspect='auto', cmap='viridis')
    ax.set_xlabel("Kapcsolási erősség (K)")
    ax.set_ylabel("Zaj szórása")
    ax.set_title("🌐 Szinkronizációs index (r) hőtérkép")
    fig.colorbar(c, ax=ax, label="r (szinkronizáció mértéke)")
    st.pyplot(fig)
