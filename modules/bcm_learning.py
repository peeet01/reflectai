import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def run():
    st.set_page_config(layout="wide")
    st.title("🧠 BCM Learning – Adaptív Szinaptikus Tanulás")

    st.markdown("""
Ez a modul a **BCM (Bienenstock–Cooper–Munro)** tanulási szabály működését modellezi, amely során a tanulás küszöbértéke időben is alkalmazkodik.

A cél: bemeneti minták alapján **stabil és dinamikusan alkalmazkodó** súlyváltozást tanulni.
    """)

    # Beállítások
    signal_type = st.sidebar.selectbox("Bemeneti jel típusa", ["Szinusz", "Fehér zaj", "Lépcsős"])
    steps = st.sidebar.slider("Szimuláció lépései", 100, 2000, 500, step=100)
    eta = st.sidebar.slider("Tanulási ráta (η)", 0.001, 0.1, 0.01)
    tau = st.sidebar.slider("Küszöb időállandó (τ)", 10, 500, 100)
    w0 = st.sidebar.slider("Kezdeti súly (w₀)", -2.0, 2.0, 0.5)
    theta0 = st.sidebar.slider("Kezdeti küszöb (θ₀)", 0.0, 1.0, 0.1)
    amplitude = st.sidebar.slider("Jel amplitúdó", 0.1, 2.0, 1.0)
    noise_level = st.sidebar.slider("Zaj szint", 0.0, 1.0, 0.0)

    # Szimuláció
    x = generate_input_signal(signal_type, steps, amplitude, noise_level)
    w, theta, y, dw, dtheta = bcm_learning(x, eta, tau, w0, theta0, steps)

    # Grafikon
    st.subheader("📈 Tanulás időfüggvényei")
    fig, ax = plt.subplots()
    ax.plot(w, label="Súly (w)")
    ax.plot(theta, label="Küszöb (θ)")
    ax.plot(y, label="Kimenet (y)")
    ax.set_title("BCM tanulási dinamika")
    ax.set_xlabel("Időlépések")
    ax.legend()
    st.pyplot(fig)

    # Export
    st.subheader("📥 Eredmények letöltése")
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "w": w,
        "θ": theta,
        "Δw": dw,
        "Δθ": dtheta
    })
    csv = df.to_csv(index_label="lépés").encode("utf-8")
    st.download_button("Letöltés CSV-ben", data=csv, file_name="bcm_learning_full.csv")

    # Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")
    st.latex(r"""
    \frac{dw}{dt} = \eta \cdot x \cdot y \cdot (y - \theta)
    \quad
    \frac{d\theta}{dt} = \frac{1}{\tau} (y^2 - \theta)
    """)
    st.markdown("""
A **BCM szabály** lehetővé teszi, hogy a neuron *dinamikusan* alkalmazkodjon a tanulás feltételeihez, nem csupán a bemenet alapján.

- A súly csak akkor nő, ha a kimenet meghaladja a küszöböt.
- A küszöb értéke is változik a kimenet függvényében (homeosztázis).
- A tanulás stabil és önszabályozó lesz – ez teszi **biológiailag relevánssá**.

**Alkalmazás:** vizuális kéreg modellezése, adaptív tanulási rendszerek, szenzoros jelfeldolgozás.
    """)

# ReflectAI kompatibilis
app = run
