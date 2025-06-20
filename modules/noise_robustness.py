import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def simulate_kuramoto(N, K, noise_level, T=100, dt=0.05):
    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)

    for _ in range(T):
        theta_diff = np.subtract.outer(theta, theta)
        interaction = np.sum(np.sin(theta_diff), axis=1)
        theta += (omega + (K / N) * interaction + noise_level * np.random.randn(N)) * dt

    r = np.abs(np.sum(np.exp(1j * theta)) / N)
    return r if np.isfinite(r) else 0.0

def run():
    st.subheader("📊 Zajtűrés és szinkronizációs robusztusság – Pro vizualizáció")

    N = st.slider("🧠 Oszcillátorok száma", 5, 100, 20)
    T = st.slider("⏱️ Iterációk száma", 50, 500, 200)
    grid_size = st.slider("📐 Rácsfelbontás", 10, 50, 30)

    K_vals = np.linspace(0, 10, grid_size)
    noise_vals = np.linspace(0, 5, grid_size)
    K_grid, Noise_grid = np.meshgrid(K_vals, noise_vals)

    R_grid = np.zeros_like(K_grid)

    with st.status("🔄 Szimuláció zajlik..."):
        for i in range(grid_size):
            for j in range(grid_size):
                r = simulate_kuramoto(N, K_vals[j], noise_vals[i], T)
                R_grid[i, j] = r

    # 🌡️ 2D hőtérkép
    fig1, ax = plt.subplots()
    heatmap = ax.pcolormesh(K_grid, Noise_grid, R_grid, shading='auto', cmap='viridis')
    cbar = fig1.colorbar(heatmap, ax=ax)
    cbar.set_label("Szinkronizációs index (r)")
    ax.set_xlabel("Kapcsolási erősség (K)")
    ax.set_ylabel("Zajszint")
    ax.set_title("🗺️ Zajtűrés – 2D hőtérkép")
    st.pyplot(fig1)

    # 🌌 3D felület
    fig2 = go.Figure(data=[go.Surface(z=R_grid, x=K_grid, y=Noise_grid, colorscale='Viridis')])
    fig2.update_layout(
        title="🎛️ Szinkronizációs robusztusság 3D felületen",
        scene=dict(
            xaxis_title='Kapcsolási erősség (K)',
            yaxis_title='Zajszint',
            zaxis_title='Szinkronizációs index (r)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig2, use_container_width=True)
