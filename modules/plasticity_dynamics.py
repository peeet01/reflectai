# modules/plasticity_dynamics.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def run():
    st.set_page_config(layout="wide")
    st.title("🧠 Hebbian és Adaptív Plaszticitás Dinamikája")

    st.markdown("""
    A tanulási szabályok leírják, hogyan módosul a szinaptikus súly a neuronok aktivitása alapján.
    Ez a modul különböző **plaszticitási modelleket** hasonlít össze interaktívan.
    """)

    # 🎛️ Paraméterek
    st.sidebar.header("⚙️ Paraméterek")
    time_steps = st.sidebar.slider("⏱️ Időlépések száma", 100, 2000, 500, 50)
    learning_rate = st.sidebar.slider("📈 Tanulási ráta (η)", 0.001, 0.1, 0.01, 0.001)
    decay = st.sidebar.slider("📉 Hanyatlás (λ)", 0.0, 0.1, 0.01, 0.005)
    noise = st.sidebar.slider("🔀 Zajszint", 0.0, 0.5, 0.1, 0.01)

    rule = st.sidebar.selectbox("🧮 Plaszticitási szabály", ["Hebbian", "Oja", "BCM"])

    # 🌱 Inicializálás
    np.random.seed(42)
    w = 0.5
    weights = [w]
    dw_values = []
    times = np.arange(time_steps)

    base_pre = np.sin(np.linspace(0, 3*np.pi, time_steps))
    base_post = np.cos(np.linspace(0, 3*np.pi, time_steps))

    pre = base_pre + noise * np.random.randn(time_steps)
    post = base_post + noise * np.random.randn(time_steps)

    # 🧪 Tanulási szabály alkalmazása
    for t in range(1, time_steps):
        if rule == "Hebbian":
            dw = learning_rate * pre[t] * post[t] - decay * w
        elif rule == "Oja":
            dw = learning_rate * (pre[t] * post[t] - post[t]**2 * w)
        elif rule == "BCM":
            theta = np.mean(post[:t])**2 if t > 1 else 0.1
            dw = learning_rate * pre[t] * post[t] * (post[t] - theta)

        w += dw
        weights.append(w)
        dw_values.append(dw)

    # 📈 Szinaptikus súlyváltozás
    st.subheader("📊 Szinaptikus súly – időbeli változás")
    fig, ax = plt.subplots()
    ax.plot(weights, label="Súly", color="tab:blue")
    ax.set_xlabel("Idő")
    ax.set_ylabel("w")
    ax.legend()
    ax.set_title("Súlydinamika")
    st.pyplot(fig)

    # 📈 Pre/Post aktivitás
    st.subheader("📈 Pre- és Post-szinaptikus aktivitás")
    fig2, ax2 = plt.subplots()
    ax2.plot(pre, label="Pre-aktivitás", color="tab:green")
    ax2.plot(post, label="Post-aktivitás", color="tab:orange")
    ax2.set_xlabel("Idő")
    ax2.set_title("Neuronaktivitások")
    ax2.legend()
    st.pyplot(fig2)

    # 🌐 3D Plot – idő/pre/súly
    st.subheader("🌐 3D Vizualizáció – Tanulási dinamika")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=times,
        y=pre,
        z=weights,
        mode='lines',
        line=dict(color=weights, colorscale='Plasma', width=3)
    )])
    fig3d.update_layout(
        scene=dict(
            xaxis_title="Idő",
            yaxis_title="Pre-aktivitás",
            zaxis_title="Súly"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # 📘 Képlet megjelenítése
    st.markdown("### 📘 Választott tanulási szabály")
    if rule == "Hebbian":
        st.latex(r"\Delta w = \eta \cdot x_{pre} \cdot x_{post} - \lambda w")
    elif rule == "Oja":
        st.latex(r"\Delta w = \eta (x_{pre} x_{post} - x_{post}^2 w)")
    elif rule == "BCM":
        st.latex(r"\Delta w = \eta \cdot x_{pre} \cdot x_{post} \cdot (x_{post} - \theta)")
        st.markdown("- ahol θ a posztszinaptikus aktivitás időátlagának négyzete")

    # 💾 CSV export
    st.subheader("💾 Eredmények letöltése")
    df = pd.DataFrame({
        "Idő": times,
        "Pre": pre,
        "Post": post,
        "Súly": weights,
        "Δw": [0] + dw_values
    })
    st.download_button("⬇️ CSV letöltése", data=df.to_csv(index=False).encode("utf-8"),
                       file_name=f"plasticity_{rule.lower()}.csv", mime="text/csv")

    # 📝 Jegyzet
    st.subheader("📝 Megfigyelések")
    st.text_area("Mit tapasztaltál különböző szabályokkal?", placeholder="Írd ide a megfigyeléseid...")

# ✅ ReflectAI-kompatibilitás
app = run
