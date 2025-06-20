import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

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
    curvature = np.dot(d_hat, cross) / (d_norm**2 + 1e-12)
    return curvature

def run():
    st.subheader("🌀 Topológiai Chern–szám és Berry-görbület – Pro változat")

    # Interaktív vezérlés
    mode = st.radio("Animáció módja:", ["🔘 Manuális léptetés", "🔄 Automatikus animáció"])
    grid_size = st.slider("Rácsfelbontás", 20, 100, 50)
    delay = st.slider("Animáció szünet (másodperc)", 0.1, 2.0, 0.4)
    mass_vals = np.linspace(-3.0, 3.0, 60)
    step = st.slider("Lépés kiválasztása", 0, len(mass_vals)-1, 0) if mode == "🔘 Manuális léptetés" else None

    # Rács létrehozása
    kx_vals = np.linspace(-np.pi, np.pi, grid_size)
    ky_vals = np.linspace(-np.pi, np.pi, grid_size)
    KX, KY = np.meshgrid(kx_vals, ky_vals)

    plot_placeholder = st.empty()

    for i, mass in enumerate(mass_vals):
        if mode == "🔘 Manuális léptetés" and i != step:
            continue

        curvature_grid = np.zeros_like(KX)
        for ix in range(grid_size):
            for iy in range(grid_size):
                curvature_grid[ix, iy] = berry_curvature(KX[ix, iy], KY[ix, iy], mass)

        chern_number = np.round(np.sum(curvature_grid) * (2 * np.pi / grid_size)**2 / (2 * np.pi), 2)
        st.markdown(f"### ℹ️ Tömegparaméter: `{mass:.2f}` | Becsült Chern-szám: `{chern_number}`")

        fig = go.Figure(data=[go.Surface(z=curvature_grid, x=KX, y=KY, colorscale='RdBu')])
        fig.update_layout(
            title="🎛️ Berry-görbület 3D – forgatható nézet",
            scene=dict(xaxis_title='kx', yaxis_title='ky', zaxis_title='Berry curvature'),
            margin=dict(l=10, r=10, b=10, t=30)
        )
        plot_placeholder.plotly_chart(fig, use_container_width=True)

        if mode == "🔄 Automatikus animáció":
            time.sleep(delay)
