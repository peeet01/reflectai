import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ✅ BCM tanulási szabály implementációja
def bcm_learning(x, eta=0.01, tau=100, w0=0.5, theta0=0.1, steps=500):
    w = w0
    theta = theta0
    w_hist, theta_hist, y_hist, dw_hist, dtheta_hist = [], [], [], [], []

    for t in range(steps):
        y = w * x[t]
        dw = eta * x[t] * y * (y - theta)
        dtheta = (y**2 - theta) / tau
        w += dw
        theta += dtheta
        w_hist.append(w)
        theta_hist.append(theta)
        y_hist.append(y)
        dw_hist.append(dw)
        dtheta_hist.append(dtheta)

    return np.array(w_hist), np.array(theta_hist), np.array(y_hist), np.array(dw_hist), np.array(dtheta_hist)

# ✅ Bemeneti jel generálás
def generate_input_signal(kind, length, amplitude=1.0, noise_level=0.0):
    t = np.linspace(0, 10, length)
    if kind == "Szinusz":
        signal = amplitude * np.sin(2 * np.pi * t)
    elif kind == "Fehér zaj":
        signal = np.random.randn(length)
    elif kind == "Lépcsős":
        signal = amplitude * np.where(t % 2 < 1, 1, 0)
    else:
        signal = np.zeros(length)
    noise = noise_level * np.random.randn(length)
    return signal + noise

# ✅ Streamlit alkalmazás
def run():
    st.set_page_config(layout="wide")
    st.title("🧠 BCM Learning – Adaptív Szinaptikus Tanulás")

    st.markdown("""
    A **BCM (Bienenstock–Cooper–Munro)** tanulási szabály egy biológiai inspirációjú modell,
    amely egy **dinamikusan változó küszöbértékkel** szabályozza, hogy mikor és mennyit tanuljon a neuron.
    """)

    # 🎛️ Beállítások
    st.sidebar.header("⚙️ Szimulációs paraméterek")
    signal_type = st.sidebar.selectbox("Bemeneti jel típusa", ["Szinusz", "Fehér zaj", "Lépcsős"])
    steps = st.sidebar.slider("Szimuláció lépései", 100, 2000, 500, step=100)
    eta = st.sidebar.slider("Tanulási ráta (η)", 0.001, 0.1, 0.01)
    tau = st.sidebar.slider("Küszöb időállandó (τ)", 10, 500, 100)
    w0 = st.sidebar.slider("Kezdeti súly (w₀)", -2.0, 2.0, 0.5)
    theta0 = st.sidebar.slider("Kezdeti küszöb (θ₀)", 0.0, 1.0, 0.1)
    amplitude = st.sidebar.slider("Jel amplitúdó", 0.1, 2.0, 1.0)
    noise_level = st.sidebar.slider("Zaj szint", 0.0, 1.0, 0.0)

    # 🔁 Szimuláció futtatása
    x = generate_input_signal(signal_type, steps, amplitude, noise_level)
    w, theta, y, dw, dtheta = bcm_learning(x, eta, tau, w0, theta0, steps)

    # 📈 2D Vizualizáció
    st.subheader("📈 Tanulási dinamika (2D)")
    fig, ax = plt.subplots()
    ax.plot(w, label="Súly (w)")
    ax.plot(theta, label="Küszöb (θ)")
    ax.plot(y, label="Kimenet (y)", linestyle='dotted')
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("Értékek")
    ax.set_title("BCM tanulás időfüggvényei")
    ax.legend()
    st.pyplot(fig)

    # 🌐 3D Vizualizáció: súly – küszöb – kimenet tér
    st.subheader("🌐 3D vizualizáció: Súly – Küszöb – Kimenet")
    st.markdown("A színek a kimeneti aktivitás (y) értékét reprezentálják – minél világosabb, annál aktívabb a neuron.")

    fig3d = go.Figure(data=[go.Scatter3d(
        x=w,
        y=theta,
        z=y,
        mode='lines+markers',
        marker=dict(size=3, color=y, colorscale='Viridis', colorbar=dict(title="Kimenet (y)")),
        line=dict(width=4, color='darkblue')
    )])

    fig3d.update_layout(
        scene=dict(
            xaxis_title="Súly (w)",
            yaxis_title="Küszöb (θ)",
            zaxis_title="Kimenet (y)"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600,
        title="Tanulási állapotok evolúciója a BCM térben"
    )

    st.plotly_chart(fig3d, use_container_width=True)

    # 📥 CSV export
    st.subheader("📥 Eredmények letöltése")
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "w": w,
        "θ": theta,
        "Δw": dw,
        "Δθ": dtheta
    })
    csv = df.to_csv(index_label="idő").encode("utf-8")
    st.download_button("⬇️ Letöltés CSV-ben", data=csv, file_name="bcm_learning_results.csv")

    # 📘 Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")
    st.latex(r"""
    \frac{dw}{dt} = \eta \cdot x \cdot y \cdot (y - \theta)
    \quad \quad
    \frac{d\theta}{dt} = \frac{1}{\tau}(y^2 - \theta)
    """)

    st.markdown("""
    - A **BCM szabály** stabilizálja a tanulást egy **homeosztatikus küszöb** segítségével.
    - A tanulás akkor aktiválódik, ha a kimeneti válasz (y) meghaladja a küszöböt (θ).
    - A tanulási ráta (η) és a küszöb időállandó (τ) szabályozzák a tanulás sebességét és stabilitását.

    **Alkalmazásai:**
    - Szenzoros kéreg modellezése (pl. látás, hallás)
    - Homeosztatikus tanulás vizsgálata
    - Neurobiológiai tanulási mechanizmusok szimulációja
    """)

# ✅ Modul regisztráció ReflectAI-kompatibilisan
app = run
