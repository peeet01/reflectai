import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# --- Matematikai függvények ---

# d-vektor
def d_vector(kx, ky, delta=0.1):
    return np.array([
        np.sin(kx),
        np.sin(ky),
        delta + np.cos(kx) + np.cos(ky)
    ])

# Berry-görbület (helyes formula)
def compute_berry_curvature(kx, ky, delta=0.1):
    d = d_vector(kx, ky, delta)
    n = np.linalg.norm(d)
    if n < 1e-10:
        return 0.0
    dkx = np.array([np.cos(kx), 0.0, -np.sin(kx)])
    dky = np.array([0.0, np.cos(ky), -np.sin(ky)])
    num = np.dot(d, np.cross(dkx, dky))
    den = n**3 + 1e-12
    return 0.5 * num / den

# Sajátvektor az alsó sávhoz
def lower_band_spinor(d):
    n = np.linalg.norm(d) + 1e-15
    v = np.array([d[0] - 1j*d[1], n - d[2]], dtype=complex)
    return v / (np.linalg.norm(v) + 1e-15)

# Berry-fázis (Wilson-loop)
def compute_berry_phase(radius=1.0, center=(0.0, 0.0), n_points=400, delta=0.1):
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    spinors = []
    for a in angles:
        kx = center[0] + radius*np.cos(a)
        ky = center[1] + radius*np.sin(a)
        spinors.append(lower_band_spinor(d_vector(kx, ky, delta)))
    spinors.append(spinors[0])  # zárjuk a hurkot
    prod = 1.0 + 0j
    for i in range(len(spinors)-1):
        prod *= np.vdot(spinors[i], spinors[i+1])
    return np.angle(prod)

# Chern-szám számítás
def compute_chern_number(delta, N=101):
    kx_vals = np.linspace(-np.pi, np.pi, N)
    ky_vals = np.linspace(-np.pi, np.pi, N)
    dk = (2*np.pi)/(N-1)
    tot = 0.0
    for kx in kx_vals:
        for ky in ky_vals:
            tot += compute_berry_curvature(kx, ky, delta)
    chern = tot * dk * dk / (2*np.pi)
    return np.round(chern, 5), chern

# Bloch-gömb pálya
def plot_3d_d_vectors(radius, center, delta=0.1):
    angles = np.linspace(0, 2*np.pi, 200)
    d_vectors = np.array([
        d_vector(center[0] + radius * np.cos(a),
                 center[1] + radius * np.sin(a),
                 delta)
        for a in angles
    ])
    norms = np.linalg.norm(d_vectors, axis=1, keepdims=True)
    d_unit = d_vectors / (norms + 1e-8)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=d_unit[:, 0], y=d_unit[:, 1], z=d_unit[:, 2],
        mode='lines+markers',
        line=dict(color='royalblue', width=3),
        marker=dict(size=2),
        name="d-hat (unit)"
    ))
    fig.update_layout(
        title="🧭 d-vektor útvonala a Bloch-gömbön",
        scene=dict(
            xaxis_title='dx', yaxis_title='dy', zaxis_title='dz',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

# --- Streamlit modul ---
def run():
    st.header("🌀 Topológiai védettség és Berry-görbület")
    st.markdown("Ez a szimuláció a 2D Brillouin-zónában vizsgálja a Berry-görbület, Berry-fázis és Chern-szám viselkedését.")

    delta = st.slider("Delta (rés paraméter)", 0.0, 2.0, 0.1, 0.01)
    N = st.slider("Pontok száma tengelyenként", 30, 150, 80, 10)

    kx_vals = np.linspace(-np.pi, np.pi, N)
    ky_vals = np.linspace(-np.pi, np.pi, N)
    curvature = np.zeros((N, N))
    curvature_data = []

    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            value = compute_berry_curvature(kx, ky, delta)
            curvature[j, i] = value
            curvature_data.append({"kx": kx, "ky": ky, "berry_curvature": value})

    # Kontúrplot
    st.subheader("📊 Berry-görbület – Kontúr ábra")
    fig, ax = plt.subplots()
    c = ax.contourf(kx_vals, ky_vals, curvature, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=ax, label="Berry-görbület")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title("Berry-görbület a Brillouin-zónában")
    st.pyplot(fig)

    # 3D ábra
    st.subheader("🌐 3D görbület vizualizáció")
    fig3d = go.Figure(data=[go.Surface(
        z=curvature,
        x=kx_vals,
        y=ky_vals,
        colorscale='RdBu',
        colorbar=dict(title='Berry curvature')
    )])
    fig3d.update_layout(title='Berry-görbület 3D felszínábra', autosize=True)
    st.plotly_chart(fig3d, use_container_width=True)

    # CSV EXPORT
    df = pd.DataFrame(curvature_data)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Görbület adatok letöltése CSV-ben", data=csv, file_name="berry_curvature.csv")

    # Berry-fázis számítás
    st.markdown("---")
    st.subheader("🧮 Berry-fázis kör mentén")
    col1, col2 = st.columns(2)
    with col1:
        radius = st.slider("Kör sugara", 0.1, 3.0, 1.0, 0.05)
    with col2:
        cx = st.slider("Középpont x", -np.pi, np.pi, 0.0)
        cy = st.slider("Középpont y", -np.pi, np.pi, 0.0)

    phase = compute_berry_phase(radius=radius, center=(cx, cy), delta=delta)
    st.success(f"Berry-fázis értéke: `{phase:.4f}` rad")

    fig_d = plot_3d_d_vectors(radius=radius, center=(cx, cy), delta=delta)
    st.plotly_chart(fig_d, use_container_width=True)

    # Chern-szám kiírás
    chern_int, chern_exact = compute_chern_number(delta)
    st.info(f"🔢 Chern-szám (numerikusan): {chern_int}  |  Pontosabb: {chern_exact:.6f}")

    # Tudományos magyarázat
    st.markdown("---")
    st.subheader("📚 Matematikai háttér")
    st.markdown(r"""
    A **Berry-görbület** egy topológiai invariáns, amelyet kvantumos rendszerek állapottér-görbületeként értelmezhetünk:  
    $$ \Omega(\mathbf k) = \frac{1}{2} \frac{\mathbf d \cdot (\partial_{k_x}\mathbf d \times \partial_{k_y}\mathbf d)}{|\mathbf d|^3} $$

    A **Berry-fázis** egy zárt görbe mentén a hullámfüggvény által szerzett geometriai fázis (Wilson-loop):  
    $$ \gamma = \oint_C \mathbf{A}(k) \cdot d\mathbf{k} $$

    A görbület integrálja egész Brillouin-zónában kvantált érték: ez a **Chern-szám**, amely meghatározza a topológiai szigetelők szélállapotainak számát.
    """)

# Kötelező ReflectAI kompatibilitás
app = run
