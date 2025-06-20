import numpy as np
import streamlit as st
import plotly.graph_objects as go

def kuramoto_with_noise(N, K, T, noise_std, dt):
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)

    for _ in range(T):
        theta_matrix = np.subtract.outer(theta, theta)
        coupling = np.sum(np.sin(theta_matrix), axis=1)
        noise = np.random.normal(0, noise_std, N)
        theta += (omega + (K / N) * coupling + noise) * dt
    r = np.abs(np.sum(np.exp(1j * theta)) / N)
    return r if np.isfinite(r) else 0.0

def run():
    st.subheader("⚡ Zajtűrés – Optimalizált Kuramoto modell")

    N = st.slider("🧠 Oszcillátorok száma", 5, 50, 20)
    T = st.slider("⏱️ Iterációk száma", 50, 300, 100)
    dt = st.slider("🔄 Időlépés (dt)", 0.01, 0.1, 0.03)

    grid_size = st.slider("📊 Rács felbontása (gyorsítás: max 10)", 3, 10, 6)

    k_min, k_max = st.slider("📡 Kapcsolási erősség (K)", 0.0, 10.0, (1.0, 3.0))
    noise_min, noise_max = st.slider("🔉 Zaj szórás", 0.0, 2.0, (0.0, 0.5))

    k_vals = np.linspace(k_min, k_max, grid_size)
    noise_vals = np.linspace(noise_min, noise_max, grid_size)
    Z = np.zeros((grid_size, grid_size))

    progress = st.progress(0)
    total = grid_size ** 2
    count = 0

    for i, noise in enumerate(noise_vals):
        for j, k in enumerate(k_vals):
            Z[i, j] = kuramoto_with_noise(N, k, T, noise, dt)
            count += 1
            progress.progress(count / total)

    fig = go.Figure(data=[go.Surface(
        z=Z, x=k_vals, y=noise_vals, colorscale='Viridis'
    )])
    fig.update_layout(
        title="🌐 Szinkronizációs index – gyors vizualizáció",
        scene=dict(
            xaxis_title='K',
            yaxis_title='Zaj (σ)',
            zaxis_title='r'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)
