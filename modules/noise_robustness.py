import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 🎯 Kuramoto szimuláció zajjal
def simulate_kuramoto(N, K, noise_level, steps=500, dt=0.05):
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(1.0, 0.1, N)
    for _ in range(steps):
        interaction = np.sum(np.sin(np.subtract.outer(theta, theta)), axis=1)
        noise = np.random.normal(0, noise_level, N)
        theta += (omega + (K / N) * interaction + noise) * dt
    return theta

# 📏 Rendparaméter

def order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))

# 🚀 Streamlit app

def run():
    st.set_page_config(layout="wide")
    st.title("📉 Zajtűrés és szinkronizáció robusztusság")

    st.markdown("""
    A **Kuramoto-modell** lehetővé teszi oszcillátorhálózatok szinkronizációjának vizsgálatát. Ebben a modulban azt figyeljük meg, 
    hogy különböző zajszintek hogyan befolyásolják a szinkronizációt – azaz a hálózat **robusztusságát** a zavaró hatásokkal szemben.
    """)

    # 🎛️ Paraméterek
    st.sidebar.header("⚙️ Paraméterek")
    N = st.sidebar.slider("Oszcillátorok száma (N)", 5, 100, 30)
    K = st.sidebar.slider("Kapcsolódási erősség (K)", 0.0, 10.0, 2.0, 0.1)
    max_noise = st.sidebar.slider("Maximális zajszint", 0.0, 2.0, 1.0, 0.05)
    steps = st.sidebar.slider("Iterációk száma", 100, 2000, 500, 100)
    dt = 0.05

    # 🔁 Szimuláció futtatása
    noise_levels = np.linspace(0, max_noise, 30)
    order_params = []
    st.subheader("🔁 Szimuláció zajlik...")

    for noise in noise_levels:
        theta = simulate_kuramoto(N, K, noise, steps=steps, dt=dt)
        r = order_parameter(theta)
        order_params.append(r)

    # 📈 2D vizualizáció
    st.subheader("📈 Szinkronizáció vs. zajszint")
    fig2d, ax = plt.subplots()
    ax.plot(noise_levels, order_params, marker="o", color='tab:blue')
    ax.set_xlabel("Zajszint")
    ax.set_ylabel("Rendparaméter (R)")
    ax.set_title("Szinkronizáció robusztussága zajjal szemben")
    ax.grid(True)
    st.pyplot(fig2d)

    # 🌐 3D Plotly vizualizáció
    st.subheader("🌐 3D vizualizáció – Zajszint vs. K vs. R")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=noise_levels,
        y=[K] * len(noise_levels),
        z=order_params,
        mode='lines+markers',
        line=dict(color=order_params, colorscale='Viridis', width=5),
        marker=dict(size=4)
    )])
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Zajszint",
            yaxis_title="K",
            zaxis_title="R (rendparaméter)"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # 💾 CSV export
    st.subheader("💾 Eredmények letöltése")
    df = pd.DataFrame({"Zajszint": noise_levels, "Rendparaméter (R)": order_params})
    st.download_button("⬇️ CSV letöltése", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="noise_robustness.csv", mime="text/csv")

    # 📘 Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")
    st.latex(r"R(t) = \left| \frac{1}{N} \sum_{j=1}^N e^{i \theta_j(t)} \right|")
    st.markdown("""
    A rendparaméter $R(t)$ a szinkronizáció mértékét jelöli:
    - $R = 1$: teljes szinkron
    - $R \approx 0$: teljes dekoherencia

    A zaj növekedésével a rendszer rendezetlensége nő, amit a $R$ csökkenése jelez. A Kuramoto-modell jól leírja a zaj és kapcsolódás közötti
    dinamikus kapcsolatot, és betekintést ad a **kollektív viselkedés stabilitásába**.
    """)

    st.subheader("📌 Konklúzió")
    st.markdown("""
    A szimuláció alapján:
    - Erősebb kapcsolatok (magas K érték) **jobban ellenállnak a zajnak**
    - Gyenge kapcsolódásnál már kis zaj is **szétzilálja** a szinkronizációt
    - A görbe lefutása alapján mérhető a **robosztusság** mértéke
    
    Ez a modell kiváló eszköz a biológiai, hálózatelméleti és komplex rendszerek **rezilienciájának** vizsgálatára.
    """)

# ✅ ReflectAI-kompatibilitás
app = run
