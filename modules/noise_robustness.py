import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

def kuramoto_with_noise(N, K, T, noise, dt=0.05):
    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2*np.pi, N)

    for _ in range(T):
        dtheta = theta[:, None] - theta
        coupling = np.sum(np.sin(dtheta), axis=1)
        theta += (omega + (K / N) * coupling) * dt
        theta += noise * np.random.normal(0, 1, N) * dt

    r = np.abs(np.sum(np.exp(1j * theta)) / N)
    return r

def run():
    st.subheader("🧪 Zajtűrés vizsgálata Kuramoto-modellel (3D vizualizációval)")

    # Paraméterek
    N = st.slider("🧠 Oszcillátorok száma", 5, 40, 20)
    T = st.slider("🕒 Iterációk száma", 50, 300, 100)
    dt = st.slider("⏱️ Időlépés (dt)", 0.01, 0.1, 0.03)

    k_start, k_end = st.slider("📡 Kapcsolási erősség tartománya (K)", 0.0, 10.0, (1.0, 3.0))
    noise_start, noise_end = st.slider("🔉 Zaj szórás tartománya", 0.0, 2.0, (0.0, 0.5))

    k_values = np.linspace(k_start, k_end, 20)
    noise_values = np.linspace(noise_start, noise_end, 20)

    R = np.zeros((len(noise_values), len(k_values)))

    # Szimuláció
    progress = st.progress(0)
    total = len(noise_values) * len(k_values)
    step = 0

    for i, noise in enumerate(noise_values):
        for j, k in enumerate(k_values):
            R[i, j] = kuramoto_with_noise(N, k, T, noise, dt)
            step += 1
            progress.progress(step / total)

    # 3D Plotly vizualizáció
    fig = go.Figure(data=[
        go.Surface(
            z=R, 
            x=k_values, 
            y=noise_values,
            colorscale="Viridis"
        )
    ])

    fig.update_layout(
        title="🌐 Szinkronizáció mértéke (r) a zaj és kapcsolási erősség függvényében",
        scene=dict(
            xaxis_title='Kapcsolási erősség (K)',
            yaxis_title='Zaj (σ)',
            zaxis_title='r-index'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    st.plotly_chart(fig, use_container_width=True)
