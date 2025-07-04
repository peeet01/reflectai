# modules/plasticity_dynamics.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def run():
    st.set_page_config(layout="wide")
    st.title("🧠 Hebbian Plaszticitás – Dinamikus Szinapszis Modell")

    st.markdown("""
    A **Hebbian-tanulás** klasszikus szabálya szerint:  
    *„neurons that fire together, wire together”*.  
    Ez a modul bemutatja a tanulási dinamika viselkedését időben különböző paraméterek mellett.
    """)

    # 🎛️ Paraméterek
    st.sidebar.header("⚙️ Szimulációs paraméterek")
    time_steps = st.sidebar.slider("⏱️ Időlépések száma", 100, 2000, 500, step=50)
    learning_rate = st.sidebar.slider("📈 Tanulási ráta (η)", 0.001, 0.1, 0.01, step=0.001)
    decay = st.sidebar.slider("📉 Hanyatlás (λ)", 0.0, 0.1, 0.01, step=0.005)
    noise_level = st.sidebar.slider("🔀 Zajszint", 0.0, 0.5, 0.1, step=0.01)

    st.subheader("🔁 Szinaptikus súlyváltozás szimuláció")

    # Inicializálás
    np.random.seed(42)
    w = 0.5
    weights = [w]
    times = list(range(time_steps))

    # Aktivitásgenerálás zajjal
    base_pre = np.sin(np.linspace(0, 3 * np.pi, time_steps))
    base_post = np.cos(np.linspace(0, 3 * np.pi, time_steps))

    pre_activity = base_pre + noise_level * np.random.randn(time_steps)
    post_activity = base_post + noise_level * np.random.randn(time_steps)

    # Súlyfrissítés
    for t in range(1, time_steps):
        dw = learning_rate * pre_activity[t] * post_activity[t] - decay * w
        w += dw
        weights.append(w)

    # 📊 Matplotlib grafikon
    fig, ax = plt.subplots()
    ax.plot(weights, label="Szinaptikus súly", color="tab:blue")
    ax.set_xlabel("Időlépés")
    ax.set_ylabel("Súlyérték")
    ax.set_title("🔁 Hebbian súlydinamika")
    ax.legend()
    st.pyplot(fig)

    # 📊 Plotly 3D vizualizáció
    st.subheader("🌐 3D Vizualizáció – Pre-aktivitás és súly időben")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=times,
        y=pre_activity,
        z=weights,
        mode='lines',
        line=dict(color=weights, colorscale='Viridis', width=3)
    )])
    fig3d.update_layout(
        scene=dict(
            xaxis_title='Idő',
            yaxis_title='Pre-aktivitás',
            zaxis_title='Szinaptikus súly'
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # 💾 CSV mentés
    st.subheader("💾 Eredmények letöltése")
    df = pd.DataFrame({
        "Idő": times,
        "Pre_aktivitás": pre_activity,
        "Post_aktivitás": post_activity,
        "Súly": weights
    })
    st.download_button("⬇️ CSV letöltése", data=df.to_csv(index=False).encode('utf-8'),
                       file_name="hebbian_dynamics.csv", mime="text/csv")

    # 🧠 Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")
    st.latex(r"""
    \Delta w = \eta \cdot x_{pre} \cdot x_{post} - \lambda \cdot w
    """)
    st.markdown("""
    - **$w$**: szinaptikus súly  
    - **$x_{pre}, x_{post}$**: pre- és posztszinaptikus neuron aktivitása  
    - **$\\eta$**: tanulási ráta  
    - **$\\lambda$**: hanyatlási konstans  
    - Az egyenlet a tanulást és a súlystabilizációt egyaránt figyelembe veszi.
    """)

    st.subheader("📝 Megfigyelések")
    st.text_area("Mit tapasztaltál a súlyváltozás vagy zaj hatására?", placeholder="Írd ide...")

# ✅ ReflectAI-kompatibilitás
app = run
