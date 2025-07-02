import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# BCM tanulási szabály implementálása
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

# Jel generátor
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

# 3D vizualizáció neuronhálóval
def draw_3d_network(weights):
    np.random.seed(42)
    num_neurons = 10
    pos = np.random.rand(num_neurons, 3)
    edges = [(i, (i + 1) % num_neurons) for i in range(num_neurons)]

    fig = go.Figure()
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[pos[i, 0], pos[j, 0]],
            y=[pos[i, 1], pos[j, 1]],
            z=[pos[i, 2], pos[j, 2]],
            mode="lines",
            line=dict(color='rgba(100,100,200,0.6)', width=2 + 4 * abs(weights[-1])),
            showlegend=False
        ))
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(size=6, color='orange'),
        name='Neuronok'
    ))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500)
    return fig

def app():
    st.title("🧠 BCM Learning – Adaptív Szinaptikus Tanulás")

    st.markdown("""
    Ez a modul a **BCM tanulási szabályt** mutatja be, amely adaptív tanulási küszöbbel egészíti ki a Hebb-elvet.
    """)

    # Paraméterek
    signal_type = st.selectbox("Bemeneti jel típusa", ["Szinusz", "Fehér zaj", "Lépcsős"])
    steps = st.slider("Szimuláció hossza (lépések)", 100, 2000, 500, step=100)
    eta = st.slider("Tanulási ráta (η)", 0.001, 0.1, 0.01, step=0.001)
    tau = st.slider("Küszöb időállandó (τ)", 10, 500, 100, step=10)

    x = generate_input_signal(signal_type, steps)
    w, theta, y = bcm_learning(x, eta, tau, steps)

    # 2D vizualizáció
    st.subheader("📈 Súly, küszöb és válasz alakulása")
    fig, ax = plt.subplots()
    ax.plot(w, label="Súly (w)")
    ax.plot(theta, label="Küszöb (θ)")
    ax.plot(y, label="Kimenet (y)")
    ax.set_xlabel("Idő")
    ax.legend()
    st.pyplot(fig)

    # 3D vizualizáció
    st.subheader("🧠 3D neuronháló")
    fig3d = draw_3d_network(w)
    st.plotly_chart(fig3d)

    # Export
    st.subheader("📥 Export")
    df = pd.DataFrame({"w": w, "theta": theta, "y": y, "x": x})
    csv = df.to_csv(index_label="idő").encode("utf-8")
    st.download_button("Letöltés CSV-ben", data=csv, file_name="bcm_learning.csv")

    # Tudományos háttér
    st.markdown("""
### 📚 Tudományos háttér

A **BCM (Bienenstock–Cooper–Munro) szabály** egy biológiailag motivált tanulási elmélet, mely szerint a szinaptikus erő **nemlineárisan** függ a posztszinaptikus aktivitástól. A tanulás során egy **adaptív küszöb** (θ) változik, amely szabályozza, hogy mikor történjen megerősítés vagy gyengítés.

**Matematikai leírás:**

- Súlyváltozás: \( \frac{dw}{dt} = \eta \cdot x \cdot y \cdot (y - \theta) \)
- Küszöbszint változása: \( \frac{d\theta}{dt} = \frac{1}{\tau}(y^2 - \theta) \)

**Értelmezés:** Ha a válasz nagyobb a küszöbnél, a szinapszis erősödik; ha kisebb, gyengül. Ez lehetővé teszi a **homeosztatikus stabilitást** és a dinamikusan szabályozott tanulást.

**Használat az appban:**
- A szinaptikus adaptációk dinamikájának modellezése
- Biológiai plaszticitás szimulálása
- Adaptív rendszerek tanulmányozása

**Felhasználás:**
- Látás- és halláskutatás
- Neurális térképek fejlődése
- Önszabályozó tanulási rendszerek tervezése
    """)
app = run
