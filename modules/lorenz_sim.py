# modules/lorenz_sim.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 🌪️ Lorenz-egyenletek
def lorenz_system(x, y, z, sigma, rho, beta):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

# 🔁 Lorenz szimuláció
def simulate_lorenz(sigma, rho, beta, dt=0.01, steps=10000, x0=0., y0=1., z0=1.05):
    x = np.empty(steps)
    y = np.empty(steps)
    z = np.empty(steps)
    x[0], y[0], z[0] = x0, y0, z0

    for i in range(1, steps):
        dx, dy, dz = lorenz_system(x[i - 1], y[i - 1], z[i - 1], sigma, rho, beta)
        x[i] = x[i - 1] + dx * dt
        y[i] = y[i - 1] + dy * dt
        z[i] = z[i - 1] + dz * dt

    return x, y, z

# 🌐 Plotly 3D attraktor
def plot_lorenz_3d(x, y, z):
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(
            width=2,
            color=np.linspace(0, 1, len(x)),
            colorscale='Turbo'
        )
    )])
    fig.update_layout(
        title="Lorenz attraktor (3D)",
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# 📉 Bifurkációs térkép
def plot_bifurcation(sigma, beta, dt, x0, y0, z0):
    st.subheader("📉 Bifurkációs térkép – változó ρ")
    rhos = np.linspace(0, 60, 400)
    x_vals = []

    for rho_val in rhos:
        x, y, z = x0, y0, z0
        # bemelegítés
        for _ in range(1000):
            dx, dy, dz = lorenz_system(x, y, z, sigma, rho_val, beta)
            x += dx * dt
            y += dy * dt
            z += dz * dt
        # gyűjtés
        for _ in range(100):
            dx, dy, dz = lorenz_system(x, y, z, sigma, rho_val, beta)
            x += dx * dt
            y += dy * dt
            z += dz * dt
            x_vals.append((rho_val, x))

    bif_rho, bif_x = zip(*x_vals)
    fig_bif, ax_bif = plt.subplots()
    ax_bif.scatter(bif_rho, bif_x, s=0.2, color='navy')
    ax_bif.set_xlabel("ρ (Rayleigh-szám)")
    ax_bif.set_ylabel("x – állapot")
    ax_bif.set_title("Lorenz bifurkációs térkép")
    st.pyplot(fig_bif)

# 🚀 App futtatása
def run():
    st.set_page_config(layout="wide")
    st.title("🌀 Lorenz-rendszer szimuláció és bifurkáció")

    st.markdown("""
A Lorenz-rendszer egy híres nemlineáris dinamikai modell, amelyet az időjárás modellezésére fejlesztettek ki,  
de azóta az egyik legismertebb **kaotikus rendszerként** vált híressé.
""")

    st.sidebar.header("⚙️ Paraméterek")
    sigma = st.sidebar.slider("σ (Prandtl-szám)", 0.0, 20.0, 10.0)
    rho = st.sidebar.slider("ρ (Rayleigh-szám)", 0.0, 60.0, 28.0)
    beta = st.sidebar.slider("β", 0.0, 10.0, 8.0 / 3.0)
    steps = st.sidebar.slider("⏱️ Iterációk száma", 1000, 20000, 10000, step=1000)
    dt = st.sidebar.number_input("🧮 Időlépés (dt)", 0.001, 0.1, 0.01, 0.001)

    if st.button("🌪️ Klasszikus Lorenz attraktor betöltése"):
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    # Szimuláció
    x, y, z = simulate_lorenz(sigma, rho, beta, dt=dt, steps=steps)
    st.subheader("🌐 3D Lorenz attraktor")
    plot_lorenz_3d(x, y, z)

    # CSV letöltés
    st.subheader("💾 Adatok letöltése")
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV letöltése", data=csv, file_name="lorenz_attractor.csv", mime="text/csv")

    # Bifurkációs térkép
    if st.checkbox("📉 Bifurkációs diagram (ρ mentén)"):
        plot_bifurcation(sigma, beta, dt, x0=0., y0=1., z0=1.05)

    # Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")

    st.latex(r"""
    \begin{aligned}
    \frac{dx}{dt} &= \sigma(y - x) \\
    \frac{dy}{dt} &= x(\rho - z) - y \\
    \frac{dz}{dt} &= xy - \beta z
    \end{aligned}
    """)

    st.markdown(r"""
**A Lorenz-rendszer** egy háromdimenziós nemlineáris differenciálegyenlet-rendszer, amely:
- **σ**: Prandtl-szám – diffúziós viszonyokat szabályozza  
- **ρ**: Rayleigh-szám – konvekciós erősség  
- **β**: geometriai paraméter

A rendszer viselkedése **bifurkációkon** megy keresztül, ha ρ értéke nő:

- $ρ < 1$: stabil fixpont
- $1 < ρ < 24.74$: oszcilláló, kvázi-periodikus állapot
- $ρ > 24.74$: **káosz**, azaz érzékenység a kezdeti feltételekre

**Bifurkációs diagram**: a rendszer hosszú távú állapotait ábrázolja egy változó paraméter (pl. ρ) mentén.  
A kaotikus viselkedés sok szórt pontként jelenik meg a diagramon.
""")

    st.subheader("📝 Megfigyelések")
    st.text_area("Mit figyeltél meg a Lorenz-rendszer szimuláció során?", placeholder="Írd ide...")

# ReflectAI-kompatibilitás
app = run
