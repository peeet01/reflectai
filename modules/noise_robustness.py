import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def simulate_kuramoto(N, K, noise_level, T=200, dt=0.05):
    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)

    for _ in range(T):
        theta_diff = np.subtract.outer(theta, theta)
        interaction = np.sum(np.sin(theta_diff), axis=1)
        noise = noise_level * np.random.randn(N)
        theta += (omega + (K / N) * interaction + noise) * dt

    r = np.abs(np.mean(np.exp(1j * theta)))
    return float(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))

def run():
    st.subheader("🎯 Zajtűrés és szinkronizációs robusztusság (stabil verzió)")

    N = st.slider("Oszcillátorok száma", 5, 100, 20)
    T = st.slider("Szimulációs idő (lépések)", 50, 500, 200)
    grid_size = st.slider("Rácsfelbontás", 10, 40, 25)

    K_vals = np.linspace(0.1, 10.0, grid_size)
    noise_vals = np.linspace(0.0, 5.0, grid_size)
    K_grid, Noise_grid = np.meshgrid(K_vals, noise_vals)

    R_grid = np.zeros_like(K_grid)

    with st.status("⏳ Számítás folyamatban..."):
        for i in range(grid_size):
            for j in range(grid_size):
                try:
                    R_grid[i, j] = simulate_kuramoto(N, K_grid[i, j], Noise_grid[i, j], T)
                except Exception:
                    R_grid[i, j] = 0.0  # Hiba esetén 0

    # 🧭 2D hőtérkép
    fig, ax = plt.subplots()
    im = ax.pcolormesh(K_grid, Noise_grid, R_grid, shading='auto', cmap='plasma')
    fig.colorbar(im, ax=ax, label="Szinkronizációs index (r)")
    ax.set_xlabel("Kapcsolási erősség (K)")
    ax.set_ylabel("Zajszint")
    ax.set_title("🔬 Zajtűrés hőtérkép")
    st.pyplot(fig)

    # 🌐 3D interaktív ábra
    fig3d = go.Figure(data=[go.Surface(
        z=R_grid,
        x=K_grid,
        y=Noise_grid,
        colorscale='Plasma'
    )])
    fig3d.update_layout(
        title="🎛️ 3D robusztussági felület",
        scene=dict(
            xaxis_title='K',
            yaxis_title='Zaj',
            zaxis_title='r'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig3d, use_container_width=True)
