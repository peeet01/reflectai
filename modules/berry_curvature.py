import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

def compute_berry_curvature(kx, ky, delta=0.1):
    d = np.array([
        np.sin(kx),
        np.sin(ky),
        delta + np.cos(kx) + np.cos(ky)
    ])
    norm = np.linalg.norm(d)
    d_hat = d / norm
    return 0.5 * d_hat[2] / (norm**2 + 1e-8)

def generate_curvature_map(N, delta):
    kx_vals = np.linspace(-np.pi, np.pi, N)
    ky_vals = np.linspace(-np.pi, np.pi, N)
    curvature = np.zeros((N, N))

    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            curvature[j, i] = compute_berry_curvature(kx, ky, delta)
    
    return kx_vals, ky_vals, curvature

def compute_chern_number(curvature, kx_vals, ky_vals):
    dkx = kx_vals[1] - kx_vals[0]
    dky = ky_vals[1] - ky_vals[0]
    integral = np.sum(curvature) * dkx * dky / (2 * np.pi)
    return integral

def run():
    st.title("🌀 Berry-görbület szimuláció")
    st.markdown("A Berry-görbület egy topológiai kvantumrendszer lokális tulajdonsága a Brillouin-zónában.")

    N = st.slider("🔢 Pontok száma tengelyenként", 30, 150, 80, 10)
    delta = st.slider("🔺 Delta paraméter (résnyitás)", -2.0, 2.0, 0.1, 0.05)
    export_csv = st.checkbox("📄 CSV export")

    kx_vals, ky_vals, curvature = generate_curvature_map(N, delta)

    # 2D Matplotlib ábra
    st.subheader("🎨 2D kontúrtérkép")
    fig, ax = plt.subplots()
    c = ax.contourf(kx_vals, ky_vals, curvature, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=ax, label="Berry-görbület")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title("Berry-görbület a Brillouin-zónában")
    st.pyplot(fig)

    # 3D Plotly térkép
    st.subheader("🌐 3D Berry-görbület")
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)
    fig3d = go.Figure(data=[go.Surface(z=curvature, x=kx_vals, y=ky_vals, colorscale="RdBu")])
    fig3d.update_layout(
        scene=dict(
            xaxis_title="kx",
            yaxis_title="ky",
            zaxis_title="Berry curvature"
        ),
        margin=dict(l=10, r=10, b=10, t=40)
    )
    st.plotly_chart(fig3d)

    # ➕ ÚJ: Chern-szám (Berry-fázis integrál)
    st.subheader("🧮 Topológiai Chern-szám")
    chern = compute_chern_number(curvature, kx_vals, ky_vals)
    st.success(f"Chern-szám ≈ `{chern:.3f}`")

    if export_csv:
        df = pd.DataFrame(curvature, index=ky_vals, columns=kx_vals)
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button("📥 Letöltés CSV-ként", data=csv, file_name="berry_curvature.csv")

    with st.expander("📘 Tudományos háttér"):
        st.markdown("""
        A **Berry-görbület** topológiai fázisokat ír le, például a kvantumos Hall-effektusban.

        A számítás az alábbi formulán alapszik:
        """)
        st.latex(r"""
        \mathbf{d}(k) = (\sin k_x, \sin k_y, \Delta + \cos k_x + \cos k_y)
        """)
        st.latex(r"""
        \Omega(k) = \frac{1}{2} \frac{d_z}{|d|^3}
        """)
        st.markdown("""
        A **Chern-szám** a teljes Brillouin-zónára integrált Berry-görbület:
        """)
        st.latex(r"""
        C = \frac{1}{2\pi} \int_{\text{BZ}} \Omega(k) \, d^2k
        """)
        st.markdown("""
        Ez a mennyiség egy egész szám (topológiai invariáns), amely meghatározza a rendszer **topológiai fázisát**.
        """)

# Kötelező ReflectAI integráció
app = run
