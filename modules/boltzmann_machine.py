import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# 🔁 Sigmoid aktiváció
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 🔄 RBM súlyfrissítés
def train_rbm(data, n_hidden, epochs, lr):
    n_visible = data.shape[1]
    weights = np.random.normal(0, 0.1, (n_visible, n_hidden))
    history = []

    for epoch in range(epochs):
        # Pozitív fázis
        pos_hidden_probs = sigmoid(np.dot(data, weights))
        pos_associations = np.dot(data.T, pos_hidden_probs)

        # Rekonstrukció
        neg_visible_probs = sigmoid(np.dot(pos_hidden_probs, weights.T))
        neg_hidden_probs = sigmoid(np.dot(neg_visible_probs, weights))
        neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

        # Súlyfrissítés
        weights += lr * (pos_associations - neg_associations.T) / data.shape[0]

        history.append(weights.copy())

    return weights, np.array(history)

# 🎛️ Felület
def run():
    st.title("🧠 Boltzmann Machine – Energetikai tanulásvizualizáció")

    st.markdown("""
    Ez a szimuláció egy **Restricted Boltzmann Machine (RBM)** egyszerűsített tanulási folyamatát mutatja be.  
    A rendszer egyenergiájú állapotokat tanul meg, és próbálja rekonstruálni a bemenetet.

    A tanulás során a súlyok változását követheted 3D vizualizációval.
    """)

    n_visible = st.slider("🔢 Látható egységek száma", 2, 10, 4)
    n_hidden = st.slider("🧠 Rejtett egységek száma", 2, 10, 3)
    epochs = st.slider("⏱️ Epoch-ok", 1, 100, 30)
    lr = st.slider("📈 Tanulási ráta", 0.001, 0.2, 0.05)

    # 🔣 Adat generálás (egyszerű bináris input)
    data = np.random.randint(0, 2, (100, n_visible))

    weights, history = train_rbm(data, n_hidden, epochs, lr)

    st.subheader("📊 3D Plotly vizualizáció a súlytér változásairól")
    frames = []

    for epoch in range(history.shape[0]):
        w = history[epoch]
        x, y, z = np.meshgrid(np.arange(n_visible), np.arange(n_hidden), [epoch])
        u = x.flatten()
        v = y.flatten()
        w_values = w.T.flatten()
        frames.append(go.Scatter3d(
            x=u,
            y=v,
            z=np.full_like(u, epoch),
            mode='markers',
            marker=dict(size=6, color=w_values, colorscale='Inferno', opacity=0.8),
            name=f"Epoch {epoch}"
        ))

    fig = go.Figure(data=frames)
    fig.update_layout(
        scene=dict(
            xaxis_title="Látható egység",
            yaxis_title="Rejtett egység",
            zaxis_title="Epoch"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )
    st.plotly_chart(fig)

    st.markdown("### 📘 Matematikai háttér")
    st.latex(r"E(v, h) = -\sum_i v_i a_i - \sum_j h_j b_j - \sum_{i,j} v_i h_j w_{ij}")
    st.markdown("""
    A Boltzmann-gép célja az **energiafüggvény minimalizálása**, ahol:
    
    - \(v_i\): látható egység állapota  
    - \(h_j\): rejtett egység állapota  
    - \(a_i, b_j\): bias értékek (itt kihagyva)  
    - \(w_{ij}\): súly a \(v_i \leftrightarrow h_j\) kapcsolatban

    **Tanulási szabály:**
    $$ \Delta w_{ij} = \epsilon (\langle v_i h_j \\rangle_{\text{data}} - \langle v_i h_j \\rangle_{\text{recon}}) $$

    - Az elv a **görbe alatti energiafelület tanulása**: az inputokat jellemző eloszlást a súlyok internalizálják.

    **Alkalmazások:**
    - Jellemzőkinyerés (feature learning)
    - Eloszlás-modellezés
    - Mély tanulási architektúrák (Deep Belief Network alapja)

    A Boltzmann Machine alapvető tanulási mintázatokat reprezentál.
    """)

    st.download_button(
        "📥 Súlytörténet letöltése (CSV)",
        data=pd.DataFrame(history[-1]).to_csv(index=False).encode("utf-8"),
        file_name="rbm_weights.csv"
    )

# App futtatás
app = run
