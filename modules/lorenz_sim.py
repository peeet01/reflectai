# modules/lorenz_sim.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# 🌪️ Lorenz-egyenletek
def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# 🔁 Lorenz szimuláció
def simulate_lorenz(sigma, rho, beta, dt=0.01, steps=10000, x0=0., y0=1., z0=1.05):
    t_span = (0, dt * steps)
    t_eval = np.linspace(*t_span, steps)
    sol = solve_ivp(lorenz_system, t_span, [x0, y0, z0], args=(sigma, rho, beta), t_eval=t_eval, method='RK45')
    return sol.t, sol.y[0], sol.y[1], sol.y[2]

# 🌐 3D Lorenz attraktor
def plot_lorenz_3d(x, y, z):
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(width=2, color=np.linspace(0, 1, len(x)), colorscale='Turbo')
    )])
    fig.update_layout(
        title="Lorenz attraktor (3D)",
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
    )
    st.plotly_chart(fig, use_container_width=True)

# 📈 Idősor plot
def plot_timeseries(t, x, y, z):
    fig, ax = plt.subplots()
    ax.plot(t, x, label="x(t)")
    ax.plot(t, y, label="y(t)")
    ax.plot(t, z, label="z(t)")
    ax.set_xlabel("Idő")
    ax.set_ylabel("Állapot")
    ax.set_title("Lorenz-idősor")
    ax.legend()
    st.pyplot(fig)

# 📉 Bifurkáció
def plot_bifurcation(sigma, beta, dt, x0, y0, z0):
    st.subheader("📉 Bifurkációs térkép – változó ρ")
    rhos = np.linspace(0, 60, 400)
    x_vals = []

    for rho in rhos:
        _, x, _, _ = simulate_lorenz(sigma, rho, beta, dt=dt, steps=2000, x0=x0, y0=y0, z0=z0)
        x_vals.extend([(rho, xi) for xi in x[-100:]])  # utolsó 100 érték

    bif_rho, bif_x = zip(*x_vals)
    fig, ax = plt.subplots()
    ax.scatter(bif_rho, bif_x, s=0.2, color='navy')
    ax.set_xlabel("ρ (Rayleigh-szám)")
    ax.set_ylabel("x – állapot")
    ax.set_title("Bifurkációs diagram")
    st.pyplot(fig)

# 🚀 App
def run():
    st.set_page_config(layout="wide")
    st.title("🌀 Lorenz-rendszer szimuláció és bifurkáció")

    st.sidebar.header("⚙️ Paraméterek")
    sigma = st.sidebar.number_input("σ", 0.0, 20.0, 10.0)
    rho = st.sidebar.number_input("ρ", 0.0, 60.0, 28.0)
    beta = st.sidebar.number_input("β", 0.0, 10.0, 8.0/3.0)
    x0 = st.sidebar.number_input("x₀", -10.0, 10.0, 0.0)
    y0 = st.sidebar.number_input("y₀", -10.0, 10.0, 1.0)
    z0 = st.sidebar.number_input("z₀", -10.0, 10.0, 1.05)
    steps = st.sidebar.slider("⏱️ Iterációk", 1000, 50000, 10000, step=1000)
    dt = st.sidebar.slider("Δt – Időlépés", 0.001, 0.1, 0.01, step=0.001)

    if st.button("🌪️ Klasszikus Lorenz paraméterek"):
        sigma, rho, beta, x0, y0, z0 = 10.0, 28.0, 8.0/3.0, 0., 1., 1.05

    t, x, y, z = simulate_lorenz(sigma, rho, beta, dt, steps, x0, y0, z0)

    st.subheader("📊 Idősor – x(t), y(t), z(t)")
    plot_timeseries(t, x, y, z)

    st.subheader("🌐 Lorenz attraktor – 3D")
    plot_lorenz_3d(x, y, z)

    st.subheader("💾 CSV letöltés")
    df = pd.DataFrame({"t": t, "x": x, "y": y, "z": z})
    st.download_button("⬇️ Letöltés", data=df.to_csv(index=False).encode("utf-8"), file_name="lorenz_attractor.csv")

    if st.checkbox("📉 Bifurkációs diagram (ρ mentén)"):
        plot_bifurcation(sigma, beta, dt, x0, y0, z0)

    st.markdown("### 📘 Tudományos háttér")
    st.latex(r'''
    \\begin{cases}
    \\frac{dx}{dt} = \\sigma (y - x) \\\\
    \\frac{dy}{dt} = x (\\rho - z) - y \\\\
    \\frac{dz}{dt} = x y - \\beta z
    \\end{cases}
    ''')
    st.markdown(r"""
A Lorenz-rendszer a káoszelmélet egyik legismertebb példája. A bifurkációs térkép segítségével vizsgálhatók a stabil és kaotikus régiók.
- **$ρ < 1$**: stabil fixpont
- **$1 < ρ < 24.74$**: oszcilláció
- **$ρ > 24.74$**: káosz
""")

    st.subheader("📝 Megfigyelések")
    st.text_area("Mit figyeltél meg?", placeholder="Írd ide...")

# ReflectAI-kompatibilitás
app = run
