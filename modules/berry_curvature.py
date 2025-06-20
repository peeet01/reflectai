import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def berry_curvature(kx, ky, mass):
    d = np.array([
        np.sin(kx),
        np.sin(ky),
        mass + np.cos(kx) + np.cos(ky)
    ])
    d_norm = np.linalg.norm(d)
    if d_norm == 0:
        return 0.0
    d_hat = d / d_norm

    # Deriváltak numerikusan (kicsi h eltolással)
    h = 1e-5
    dkx = np.array([
        np.cos(kx),
        0,
        -np.sin(kx)
    ])
    dky = np.array([
        0,
        np.cos(ky),
        -np.sin(ky)
    ])
    cross = np.cross(dkx, dky)
    curvature = np.dot(d_hat, cross) / (d_norm ** 2 + 1e-12)
    return curvature

def run():
    st.subheader("🌀 Topológiai Chern–szám és Berry-görbület vizualizáció (Pro)")

    # Paraméterek
    grid_size = st.slider("Rácsfelbontás", 20, 100, 50)
    mass = st.slider("Topológiai tömeg (mass paraméter)", -5.0, 5.0, 1.0)

    # Rács
    kx_vals = np.linspace(-np.pi, np.pi, grid_size)
    ky_vals = np.linspace(-np.pi, np.pi, grid_size)
    KX, KY = np.meshgrid(kx_vals, ky_vals)

    curvature_grid = np.zeros_like(KX)

    # Berry görbület kiszámítása
    for i in range(grid_size):
        for j in range(grid_size):
            curvature_grid[i, j] = berry_curvature(KX[i, j], KY[i, j], mass)

    chern_number = np.round(np.sum(curvature_grid) * (2 * np.pi / grid_size)**2 / (2 * np.pi), 2)

    st.markdown(f"### 🌐 Becsült Chern-szám: `{chern_number}`")

    # 2D hőtérkép
    fig1, ax = plt.subplots()
    c = ax.pcolormesh(KX, KY, curvature_grid, cmap='RdBu', shading='auto')
    fig1.colorbar(c, ax=ax, label="Berry görbület")
    ax.set_title("🗺️ 2D Berry görbület térkép")
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    st.pyplot(fig1)

    # 3D Plotly ábra
    fig2 = go.Figure(data=[go.Surface(z=curvature_grid, x=KX, y=KY, colorscale='RdBu')])
    fig2.update_layout(title="🌌 3D Berry görbület felület",
                       scene=dict(xaxis_title='kx', yaxis_title='ky', zaxis_title='Berry curvature'),
                       margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig2, use_container_width=True)
