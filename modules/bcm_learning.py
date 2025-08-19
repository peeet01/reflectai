import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ✅ BCM tanulási szabály implementációja (dt-vel, klippeléssel)
def bcm_learning(x, eta=0.01, tau=100, w0=0.5, theta0=0.1, steps=500, dt=1.0,
                 w_clip=None, theta_min=0.0):
    w = w0
    theta = theta0
    w_hist, theta_hist, y_hist, dw_hist, dtheta_hist = [], [], [], [], []

    for t in range(steps):
        y = w * x[t]
        dw = eta * x[t] * y * (y - theta)       # Δw
        dtheta = (y**2 - theta) / tau           # Δθ

        # diszkrét lépés dt-vel
        w += dt * dw
        theta += dt * dtheta

        # opcionális klippelés / korlátok
        if w_clip is not None:
            w = float(np.clip(w, -w_clip, w_clip))
        if theta_min is not None:
            theta = float(max(theta, theta_min))

        w_hist.append(w)
        theta_hist.append(theta)
        y_hist.append(y)
        dw_hist.append(dw)
        dtheta_hist.append(dtheta)

    return (np.array(w_hist), np.array(theta_hist), np.array(y_hist),
            np.array(dw_hist), np.array(dtheta_hist))

# ✅ Bemeneti jel generálás
def generate_input_signal(kind, length, amplitude=1.0, noise_level=0.0, seed=None,
                          standardize=False):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, length)

    if kind == "Szinusz":
        signal = amplitude * np.sin(2 * np.pi * t)
    elif kind == "Fehér zaj":
        signal = rng.standard_normal(length) * amplitude
    elif kind == "Lépcsős":
        signal = amplitude * np.where(t % 2 < 1, 1.0, 0.0)
    else:
        signal = np.zeros(length)

    noise = noise_level * rng.standard_normal(length)
    x = signal + noise

    if standardize:
        s = np.std(x)
        if s > 0:
            x = (x - np.mean(x)) / s

    return x

# ✅ Streamlit alkalmazás
def run():
    st.set_page_config(layout="wide")
    st.title("🧠 BCM Learning – Adaptív Szinaptikus Tanulás")

    st.markdown("""
    A **BCM (Bienenstock–Cooper–Munro)** tanulási szabály egy dinamikus küszöbbel szabályozza,
    hogy mikor erősödik/gyengül a szinapszis. Pozitív *y−θ* mellett **LTP**, negatívnál **LTD**.
    """)

    # 🎛️ Beállítások
    st.sidebar.header("⚙️ Szimulációs paraméterek")
    signal_type = st.sidebar.selectbox("Bemeneti jel típusa", ["Szinusz", "Fehér zaj", "Lépcsős"])
    steps = st.sidebar.slider("Szimuláció lépései", 100, 5000, 500, step=100)
    eta = st.sidebar.slider("Tanulási ráta (η)", 0.0005, 0.1, 0.01)
    tau = st.sidebar.slider("Küszöb időállandó (τ)", 10, 1000, 100)
    dt = st.sidebar.slider("Időlépés (dt)", 0.1, 2.0, 1.0, 0.1)

    w0 = st.sidebar.slider("Kezdeti súly (w₀)", -2.0, 2.0, 0.5)
    theta0 = st.sidebar.slider("Kezdeti küszöb (θ₀)", 0.0, 1.0, 0.1)
    amplitude = st.sidebar.slider("Jel amplitúdó", 0.1, 3.0, 1.0)
    noise_level = st.sidebar.slider("Zaj szint", 0.0, 1.5, 0.0)
    standardize = st.sidebar.checkbox("Standardizált bemenet (z-score)", value=False)

    w_clip_on = st.sidebar.checkbox("Súly klippelése", value=True)
    w_clip = st.sidebar.slider("Klippelt súly |w| ≤", 0.5, 10.0, 5.0) if w_clip_on else None
    theta_min = st.sidebar.slider("Minimum küszöb θ ≥", 0.0, 1.0, 0.0)

    seed = st.sidebar.number_input("Véletlen mag (seed)", value=42, step=1)

    # ▶️ futtatás gombbal
    if st.button("🚀 Szimuláció futtatása"):
        with st.spinner("Számolás…"):
            x = generate_input_signal(signal_type, steps, amplitude, noise_level,
                                      seed=int(seed), standardize=standardize)
            w, theta, y, dw, dtheta = bcm_learning(
                x, eta, tau, w0, theta0, steps, dt, w_clip, theta_min
            )

        # 📈 2D Vizualizáció
        st.subheader("📈 Tanulási dinamika (2D)")
        fig, ax = plt.subplots()
        ax.plot(w, label="Súly (w)")
        ax.plot(theta, label="Küszöb (θ)")
        ax.plot(y, label="Kimenet (y)", linestyle='dotted')
        ax.set_xlabel("Időlépések")
        ax.set_ylabel("Értékek")
        ax.set_title("BCM tanulás időfüggvényei")
        ax.legend(loc="best")
        st.pyplot(fig)

        # 🌐 3D Vizualizáció
        st.subheader("🌐 3D vizualizáció: Súly – Küszöb – Kimenet")
        st.markdown("A színek a kimeneti aktivitás (y) értékét jelzik.")

        fig3d = go.Figure(data=[go.Scatter3d(
            x=w, y=theta, z=y,
            mode='lines+markers',
            marker=dict(size=3, color=y, colorscale='Viridis',
                        colorbar=dict(title="Kimenet (y)")),
            line=dict(width=4)
        )])
        fig3d.update_layout(
            scene=dict(xaxis_title="Súly (w)", yaxis_title="Küszöb (θ)", zaxis_title="Kimenet (y)"),
            margin=dict(l=0, r=0, b=0, t=40),
            height=600,
            title="Tanulási állapotok evolúciója a BCM térben"
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # 📥 CSV export
        st.subheader("📥 Eredmények letöltése")
        df = pd.DataFrame({
            "x": x, "y": y, "w": w, "theta": theta, "dw": dw, "dtheta": dtheta
        })
        csv = df.to_csv(index_label="time_step").encode("utf-8")
        st.download_button("⬇️ Letöltés CSV-ben",
                           data=csv, file_name="bcm_learning_results.csv")

        # 📘 Háttér
        st.markdown("### 📘 Tudományos háttér")
        st.latex(r"""
        \Delta w = \eta \, x\, y \,(y-\theta),
        \qquad
        \Delta \theta = \frac{1}{\tau}\,(y^2-\theta)
        """)
        st.markdown("""
        - **Stabilitási tippek:** nagy **η** vagy kicsi **τ** → érdemes csökkenteni **dt**-t, vagy bekapcsolni a **klippelést**.  
        - A **standardizált bemenet** csökkenti a divergencia esélyét zajos jel esetén.
        """)

# ✅ Modul regisztráció ReflectAI-kompatibilisen
app = run
