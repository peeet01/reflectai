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

# 📈 Idősor megjelenítés
def plot_timeseries(x, y, z, dt):
    t = np.arange(len(x)) * dt
    fig, ax = plt.subplots()
    ax.plot(t, x, label='x(t)')
    ax.plot(t, y, label='y(t)')
    ax.plot(t, z, label='z(t)')
    ax.set_title("Lorenz-idősor")
    ax.set_xlabel("Idő")
    ax.set_ylabel("Állapotváltozók")
    ax.legend()
    st.pyplot(fig)

# 📉 Bifurkációs térkép
def plot_bifurcation(sigma, beta, dt, x0, y0, z0):
    st.subheader("📉 Bifurkációs térkép – változó ρ")
    rhos = np.linspace(0, 60, 400)
    x_vals = []

    for rho_val in rhos:
        x, y, z = x0, y0, z0
        for _ in range(1000):  # bemelegítés
            dx, dy, dz = lorenz_system(x, y, z, sigma, rho_val, beta)
            x += dx * dt
            y += dy * dt
            z += dz * dt
        for _ in range(100):  # gyűjtés
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

# 🚀 Streamlit app
def run():
    st.set_page_config(layout="wide")
    st.title("🌀 Lorenz-rendszer szimuláció és bifurkáció")

    st.markdown("""
A **Lorenz-rendszer** három differenciálegyenletből álló nemlineáris rendszer,  
amely **determinista káosz** tanulmányozására szolgál.  
Vizsgáljuk meg a fázistérbeli trajektóriákat és időbeli viselkedést.
""")

    # 🌐 Paraméterek
    st.sidebar.header("⚙️ Paraméterek")
    sigma = st.sidebar.number_input("σ (Prandtl-szám)", 0.0, 20.0, 10.0)
    rho = st.sidebar.number_input("ρ (Rayleigh-szám)", 0.0, 60.0, 28.0)
    beta = st.sidebar.number_input("β", 0.0, 10.0, 8.0 / 3.0)
    x0 = st.sidebar.number_input("x₀", -10.0, 10.0, 0.0)
    y0 = st.sidebar.number_input("y₀", -10.0, 10.0, 1.0)
    z0 = st.sidebar.number_input("z₀", -10.0, 10.0, 1.05)
    steps = st.sidebar.slider("⏱️ Iterációk száma", 1000, 50000, 10000, step=1000)
    dt = st.sidebar.slider("Δt – Időlépés", 0.001, 0.1, 0.01, step=0.001)

    if st.button("🌪️ Klasszikus Lorenz attraktor betöltése"):
        sigma, rho, beta, x0, y0, z0 = 10.0, 28.0, 8.0/3.0, 0., 1., 1.05

    # Szimuláció
    x, y, z = simulate_lorenz(sigma, rho, beta, dt=dt, steps=steps, x0=x0, y0=y0, z0=z0)

    # 📈 Idősor
    st.subheader("📊 Idősor – x(t), y(t), z(t)")
    plot_timeseries(x, y, z, dt)

    # 🌐 3D attraktor
    st.subheader("🌐 Lorenz attraktor – 3D")
    plot_lorenz_3d(x, y, z)

    # 💾 CSV letöltés
    st.subheader("💾 Adatok letöltése")
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV letöltése", data=csv, file_name="lorenz_attractor.csv", mime="text/csv")

    # 📉 Bifurkációs térkép
    if st.checkbox("📉 Bifurkációs diagram (ρ mentén)"):
        plot_bifurcation(sigma, beta, dt, x0, y0, z0)

    # 📘 Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")
    st.latex(r"""
    \begin{cases}
    \frac{dx}{dt} = \sigma (y - x) \\
    \frac{dy}{dt} = x (\rho - z) - y \\
    \frac{dz}{dt} = x y - \beta z
    \end{cases}
    """)

    st.markdown(r"""
A Lorenz-rendszer egy híres **nemlineáris determinisztikus** rendszer, amely erősen **érzékeny a kezdeti feltételekre**.  
A dinamikája a paraméterek függvényében drasztikusan változhat:

- **$ρ < 1$**: stabil fixpont  
- **$1 < ρ < 24.74$**: oszcilláló, kvázi-periodikus állapot  
- **$ρ > 24.74$**: kaotikus attraktor – Lorenz pillangó

A **bifurkációs térkép** segítségével vizualizálható, mikor alakul ki **stabilitás vagy káosz** a rendszerben.

**Alkalmazások**:
- Meteorológiai modellezés  
- Káoszelmélet  
- Nemlineáris rendszerek oktatása
""")

    st.subheader("📝 Megfigyelések")
    st.text_area("Mit figyeltél meg a Lorenz-rendszer szimuláció során?", placeholder="Írd ide...")

# ✅ ReflectAI-kompatibilitás
app = run
