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
    st.subheader("📊 Zajtűrés vizsgálata Kuramoto-modellel – Pro vizualizáció")

    # Beállítható paraméterek
    N = st.slider("🧠 Oszcillátorok száma", 5, 50, 20)
    T = st.slider("⏱️ Iterációk száma", 100, 300, 150)
    dt = st.slider("🔄 Időlépés (dt)", 0.01, 0.1, 0.03)

    k_min, k_max = st.slider("📡 Kapcsolási erősség (K) tartomány", 0.0, 10.0, (1.0, 3.0))
    noise_min, noise_max = st.slider("🔉 Zaj szórás tartomány", 0.0, 2.0, (0.0, 0.5))

    k_vals = np.linspace(k_min, k_max, 15)
    noise_vals = np.linspace(noise_min, noise_max, 15)

    Z = np.zeros((len(noise_vals), len(k_vals)))

    progress = st.progress(0)
    total = len(noise_vals) * len(k_vals)
    step = 0

    for i, noise in enumerate(noise_vals):
        for j, k in enumerate(k_vals):
            Z[i, j] = kuramoto_with_noise(N, k, T, noise, dt)
            step += 1
            progress.progress(step / total)

    fig = go.Figure(data=[
        go.Surface(
            z=Z,
            x=k_vals,
            y=noise_vals,
            colorscale='Viridis'
        )
    ])
    fig.update_layout(
        title="🌐 Szinkronizációs index (r) – zaj és K függvényében",
        scene=dict(
            xaxis_title='K',
            yaxis_title='Zaj szórás (σ)',
            zaxis_title='Szinkronizáció (r)'
        ),
        margin=dict(l=10, r=10, b=10, t=50)
    )
    st.plotly_chart(fig, use_container_width=True)
