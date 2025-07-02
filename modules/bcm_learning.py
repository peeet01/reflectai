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

# Új 3D neuronháló vizualizáció (minden súly szerepel)
def draw_3d_network(w_array):
    np.random.seed(0)
    N = len(w_array)
    pos = np.random.rand(N, 3) * 10

    fig = go.Figure()

    def get_weight_color(weight):
        norm = np.clip((abs(weight) - 0.1) / 0.5, 0, 1)
        r = int(255 * (1 - norm))
        g = int(100 * norm)
        b = int(255 * norm)
        return f'rgb({r},{g},{b})'

    for i in range(N - 1):
        fig.add_trace(go.Scatter3d(
            x=[pos[i, 0], pos[i + 1, 0]],
            y=[pos[i, 1], pos[i + 1, 1]],
            z=[pos[i, 2], pos[i + 1, 2]],
            mode="lines",
            line=dict(
                color=get_weight_color(w_array[i]),
                width=1.5 + 3 * abs(w_array[i])
            ),
            hoverinfo="none",
            showlegend=False
        ))

    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="markers+text",
        marker=dict(size=6, color="orange"),
        text=[f"w={w:.2f}" for w in w_array],
        hoverinfo="text",
        name="Neuronok"
    ))

    fig.update_layout(scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ), height=500, margin=dict(l=0, r=0, b=0, t=40))

    return fig

# Fő futtatófüggvény
def run():
    st.title("🧠 BCM Learning – Adaptív Szinaptikus Tanulás")

    st.markdown("""
Ez a modul a **BCM (Bienenstock–Cooper–Munro)** tanulási szabály működését szemlélteti, amely a szinaptikus módosulásokat egy dinamikusan változó küszöbön keresztül modellezi.
    """)

    signal_type = st.selectbox("Bemeneti jel típusa", ["Szinusz", "Fehér zaj", "Lépcsős"])
    steps = st.slider("Szimuláció lépései", 100, 2000, 500, step=100)
    eta = st.slider("Tanulási ráta (η)", 0.001, 0.1, 0.01, step=0.001)
    tau = st.slider("Küszöb időállandó (τ)", 10, 500, 100, step=10)

    x = generate_input_signal(signal_type, steps)
    w, theta, y = bcm_learning(x, eta, tau, steps)

    # 2D vizualizáció
    st.subheader("📈 Tanulási dinamika")
    fig, ax = plt.subplots()
    ax.plot(w, label="Súly (w)")
    ax.plot(theta, label="Küszöb (θ)")
    ax.plot(y, label="Kimenet (y)")
    ax.set_xlabel("Idő")
    ax.set_title("BCM súlytanulás dinamikája")
    ax.legend()
    st.pyplot(fig)

    # 3D vizualizáció
    st.subheader("🔬 3D neuronháló vizualizáció")
    st.plotly_chart(draw_3d_network(w))

    # Export
    st.subheader("📥 Eredmények letöltése")
    df = pd.DataFrame({"w": w, "θ": theta, "y": y, "x": x})
    csv = df.to_csv(index_label="idő").encode("utf-8")
    st.download_button("Letöltés CSV-ben", data=csv, file_name="bcm_learning.csv")

    # Tudományos háttér
    st.markdown("""
### 📚 Tudományos háttér

A **BCM-szabály** a szinaptikus plaszticitás egyik biológiailag megalapozott modellje, amely egy **nemlineáris aktivitásfüggő** tanulási küszöböt (θ) használ.

**Formális leírás:**

- Súlyváltozás:  
  \( \frac{dw}{dt} = \eta \cdot x \cdot y \cdot (y - \theta) \)

- Küszöbszint változása:  
  \( \frac{d\theta}{dt} = \frac{1}{\tau} (y^2 - \theta) \)

**Jelentőség:**

- Homeosztatikus stabilitást biztosít  
- Szelektív tanulást tesz lehetővé  
- Biológiailag releváns: szenzoros plaszticitás, látásrendszer fejlődése stb.

**Használat az appban:**

- Szinaptikus tanulás időbeli dinamikájának vizsgálata  
- Vizualizáció neurális kapcsolatok erősödéséről és gyengüléséről  
- Interaktív kísérletezés eltérő bemeneti jelekkel
    """)

# ❗ Fontos a kompatibilitás miatt
app = run
