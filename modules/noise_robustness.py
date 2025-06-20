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
    return r

def run():
    st.subheader("🧪 Zajtűrés vizsgálata Kuramoto-modellel (Optimalizált 3D)")

    # Paraméterek
    N = st.slider("🧠 Oszcillátorok száma", 5, 30, 15)
    T = st.slider("⏱️ Iterációk száma", 50, 200, 80)
    dt = st.slider("🔄 Időlépés (dt)", 0.01, 0.1, 0.03)

    k_start, k_end = st.slider("📡 Kapcsolási erősség tartománya (K)", 0.0, 10.0, (1.0, 3.0))
    noise_start, noise_end = st.slider("🔉 Zaj szórás tartománya", 0.0, 2.0, (0.0, 0.5))

    k_values = np.linspace(k_start, k_end, 10)
    noise_values = np.linspace(noise_start, noise_end, 10)

    R = np.zeros((len(noise_values), len(k_values)))

    progress = st.progress(0)
    total = len(noise_values) * len(k_values)
    step = 0

    for i, noise in enumerate(noise_values):
        for j, k in enumerate(k_values):
            try:
                r_value = kuramoto_with_noise(N, k, T, noise, dt)
                if np.isnan(r_value) or np.isinf(r_value):
                    r_value = 0
                R[i, j] = r_value
            except:
                R[i, j] = 0
            step += 1
            progress.progress(step / total)

    fig = go.Figure(data=[
        go.Surface(
            z=R,
            x=k_values,
            y=noise_values,
            colorscale='Viridis'
        )
    ])
    fig.update_layout(
        title="🌐 Szinkronizáció mértéke (r) a zaj és K függvényében",
        scene=dict(
            xaxis_title='Kapcsolási erősség (K)',
            yaxis_title='Zaj (σ)',
            zaxis_title='Szinkronizáció (r)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    st.plotly_chart(fig, use_container_width=True)
