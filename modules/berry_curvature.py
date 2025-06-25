import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def compute_berry_curvature(kx, ky):
    """Egyszerű Berry-görbület szimuláció 2D Brillouin-zónában."""
    # Példa egy Dirac-szerű Hamilton operátorra
    delta = 0.1
    d = np.array([
        np.sin(kx),
        np.sin(ky),
        delta + np.cos(kx) + np.cos(ky)
    ])
    norm = np.linalg.norm(d)
    d_hat = d / norm
    # Berry-görbület numerikus közelítése
    return 0.5 * d_hat[2] / (norm**2 + 1e-8)

def run():
    st.header("🌀 Topológiai védettség és Berry-görbület")
    st.markdown("Ez a szimuláció a 2D Brillouin-zónában vizsgálja a Berry-görbület eloszlását.")

    N = st.slider("Pontok száma tengelyenként", min_value=30, max_value=150, value=80, step=10)

    kx_vals = np.linspace(-np.pi, np.pi, N)
    ky_vals = np.linspace(-np.pi, np.pi, N)
    curvature = np.zeros((N, N))

    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            curvature[j, i] = compute_berry_curvature(kx, ky)

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(kx_vals, ky_vals, curvature, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=ax, label="Berry-görbület")
    ax.set_title("Berry-görbület a Brillouin-zónában")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    st.pyplot(fig)
    app = run
