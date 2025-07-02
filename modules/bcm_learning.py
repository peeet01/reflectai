import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# BCM tanulási szabály
def bcm_learning(x, eta=0.01, tau=100, steps=500):
    w = 0.5
    theta = 0.1
    w_hist, theta_hist, y_hist = [], [], []

    for t in range(steps):
        y = w * x[t]
        dw = eta * x[t] * y * (y - theta)
        dtheta = (y**2 - theta) / tau
        w += dw
        theta += dtheta
        w_hist.append(w)
        theta_hist.append(theta)
        y_hist.append(y)

    return np.array(w_hist), np.array(theta_hist), np.array(y_hist)

# Jelgenerátor
def generate_input_signal(kind, length):
    t = np.linspace(0, 10, length)
    if kind == "Szinusz":
        return np.sin(2 * np.pi * t)
    elif kind == "Fehér zaj":
        return np.random.randn(length)
    elif kind == "Lépcsős":
        return np.where(t % 2 < 1, 1, 0)
    else:
        return np.zeros(length)

# 🔁 Új 3D neuronháló vizualizáció
def draw_3d_network(weight):
    np.random.seed(42)
    num_nodes = 8
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    radius = 1

    # Kör alakú elrendezés
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.sin(2 * angles)

    fig = go.Figure()

    # Élek
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[j]],
            y=[y[i], y[j]],
            z=[z[i], z[j]],
            mode='lines',
            line=dict(
                width=2 + abs(weight) * 8,
                color='rgba(0, 150, 255, 0.6)'
            ),
            showlegend=False
        ))

    # Csomópontok
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=6,
            color=weight,
            colorscale='Plasma',
            colorbar=dict(title="Súly"),
            showscale=True
        ),
        name='Neuronok'
    ))

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        title="3D Neuronháló súlymegjelenítéssel"
    )

    return fig

# ✅ App futtatása
def run():
    st.title("🧠 BCM Learning – Adaptív Szinaptikus Tanulás")

    st.markdown("""
Ez a modul a **BCM (Bienenstock–Cooper–Munro)** tanulási szabály működését szemlélteti, amely a szinaptikus módosulásokat egy dinamikusan változó küszöbön keresztül modellezi.
    """)

    # Paraméterek
    signal_type = st.selectbox("Bemeneti jel típusa", ["Szinusz", "Fehér zaj", "Lépcsős"])
    steps = st.slider("Szimuláció lépései", 100, 2000, 500, step=100)
    eta = st.slider("Tanulási ráta (η)", 0.001, 0.1, 0.01, step=0.001)
    tau = st.slider("Küszöb időállandó (τ)", 10, 500, 100, step=10)

    x = generate_input_signal(signal_type, steps)
    w, theta, y = bcm_learning(x, eta, tau, steps)

    # 2D ábrák
    st.subheader("📈 Tanulási dinamika")
    fig, ax = plt.subplots()
    ax.plot(w, label="Súly (w)")
    ax.plot(theta, label="Küszöb (θ)")
    ax.plot(y, label="Kimenet (y)")
    ax.set_xlabel("Idő")
    ax.set_title("BCM súlytanulás dinamikája")
    ax.legend()
    st.pyplot(fig)

    # 3D neuronháló
    st.subheader("🔬 3D neuronháló vizualizáció")
    st.plotly_chart(draw_3d_network(w[-1]))

    # CSV export
    st.subheader("📥 Eredmények letöltése")
    df = pd.DataFrame({"w": w, "θ": theta, "y": y, "x": x})
    csv = df.to_csv(index_label="idő").encode("utf-8")
    st.download_button("Letöltés CSV-ben", data=csv, file_name="bcm_learning.csv")

    # Tudományos háttér
    st.markdown("""
### 📚 Tudományos háttér

A **BCM-szabály** egy biológiailag inspirált tanulási modell, amely nemlineáris módon módosítja a szinapszisokat egy dinamikusan változó küszöb alapján.

**Formális leírás:**

- Súlyváltozás:  
  \( \frac{dw}{dt} = \eta \cdot x \cdot y \cdot (y - \theta) \)

- Küszöbszint változása:  
  \( \frac{d\theta}{dt} = \frac{1}{\tau} (y^2 - \theta) \)

**Jelentőség:**
- Homeosztatikus tanulás
- Adaptív válaszküszöb
- Szenzoros rendszer fejlődésének modellezése

**Alkalmazás a modulban:**
- Interaktív jel-vezérelt tanulás
- Vizualizáció neurális válaszokra és súlymódosulásra
- Biológiai ihletésű tanulás demonstrálása
    """)

# 🔁 Kötelező: modul kompatibilitás
app = run
