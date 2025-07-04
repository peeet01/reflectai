import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 🌪️ Lorenz egyenletek
def lorenz_system(x, y, z, sigma, rho, beta):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

# 🚀 Streamlit app
def run():
    st.set_page_config(layout="wide")
    st.title("🌪️ Lorenz attraktor – Kaotikus dinamikus rendszer")

    st.markdown("""
A **Lorenz-rendszer** három differenciálegyenletből álló nemlineáris rendszer,  
amely **determinista káosz** tanulmányozására szolgál.  
Vizsgáljuk meg a fázistérbeli trajektóriákat és idősorokat.
    """)

    # 🎛️ Paraméterek
    st.sidebar.header("⚙️ Szimulációs paraméterek")

    sigma = st.sidebar.slider("σ (Prandtl-szám)", 0.0, 20.0, 10.0, 0.1)
    rho = st.sidebar.slider("ρ (Rayleigh-szám)", 0.0, 50.0, 28.0, 0.5)
    beta = st.sidebar.slider("β", 0.0, 10.0, 8.0 / 3.0, 0.05)

    x0 = st.sidebar.number_input("x₀", value=0.0)
    y0 = st.sidebar.number_input("y₀", value=1.0)
    z0 = st.sidebar.number_input("z₀", value=1.05)

    steps = st.sidebar.slider("⏱️ Iterációk száma", 1000, 20000, 10000, 1000)
    dt = st.sidebar.slider("Δt – Időlépés", 0.001, 0.1, 0.01, 0.001)

    use_plotly = st.sidebar.checkbox("🎨 Plotly használata (3D)", value=False)

    # 🔄 Szimuláció
    x = np.empty(steps)
    y = np.empty(steps)
    z = np.empty(steps)
    t = np.linspace(0, steps * dt, steps)
    x[0], y[0], z[0] = x0, y0, z0

    for i in range(1, steps):
        dx, dy, dz = lorenz_system(x[i - 1], y[i - 1], z[i - 1], sigma, rho, beta)
        x[i] = x[i - 1] + dx * dt
        y[i] = y[i - 1] + dy * dt
        z[i] = z[i - 1] + dz * dt

    # 📈 Idősoros lebontás
    st.subheader("📊 Idősor – x(t), y(t), z(t)")
    fig_ts, ax_ts = plt.subplots()
    ax_ts.plot(t, x, label='x(t)')
    ax_ts.plot(t, y, label='y(t)')
    ax_ts.plot(t, z, label='z(t)')
    ax_ts.set_xlabel("Idő")
    ax_ts.set_ylabel("Állapotváltozók")
    ax_ts.legend()
    ax_ts.set_title("Lorenz-idősor")
    st.pyplot(fig_ts)

    # 🌐 3D trajektória
    st.subheader("🌀 Lorenz attraktor 3D-ben")

    if use_plotly:
        fig3d = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=np.linspace(0, 1, len(x)), colorscale='Turbo', width=2)
        ))
        fig3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=30),
            scene=dict(
                xaxis_title='x', yaxis_title='y', zaxis_title='z',
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                zaxis=dict(showticklabels=False)
            )
        )
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, lw=0.5)
        ax.set_title("Lorenz attraktor")
        st.pyplot(fig)

    # 💾 CSV export
    st.subheader("💾 Adatok letöltése")
    df = pd.DataFrame({"t": t, "x": x, "y": y, "z": z})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV letöltése", data=csv, file_name="lorenz_trajectory.csv", mime="text/csv")

    # 📘 Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")

    st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \sigma (y - x) \\
\frac{dy}{dt} = x (\rho - z) - y \\
\frac{dz}{dt} = xy - \beta z
\end{cases}
""")

    st.markdown("""
- A Lorenz-rendszer **nemlineáris determinisztikus rendszer**, amely erősen érzékeny a kezdeti feltételekre.
- A paraméterek beállításával különböző dinamikai viselkedések (fixpont, ciklus, attraktor) jelenhetnek meg.
- A klasszikus káoszos állapot: σ = 10, ρ = 28, β = 8/3

#### Alkalmazások:
- Meteorológia
- Káosz-elmélet
- Nemlineáris rendszerek oktatása
""")

    # 📝 Megfigyelések
    st.subheader("📝 Megfigyelések")
    st.text_area("Mit tapasztaltál az attraktor viselkedésében?", placeholder="Írd ide...")

# ReflectAI kompatibilitás
app = run
