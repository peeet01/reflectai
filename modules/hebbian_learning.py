import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 🔁 Hebbian tanulási algoritmus
def hebbian_learning(X, T, eta, epochs):
    weights = np.zeros(X.shape[1])
    history = []
    for _ in range(epochs):
        for x, t in zip(X, T):
            weights += eta * x * t
            history.append(weights.copy())
    return np.array(history)

# 🎯 Bemeneti adatok (AND logika)
def generate_inputs():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    T = np.array([0, 0, 0, 1])
    return X, T

# 🚀 Streamlit app
def run():
    st.title("🧠 Hebbian Learning – Egyszerű szinaptikus tanulás")

    st.markdown("""
    A Hebbian tanulás egy alapvető tanulási szabály, amely az agyban zajló **szinaptikus plaszticitást** modellezi.  
    A tanulási folyamat során a súlyok módosulása attól függ, hogy a bemenet és a kimenet **egyszerre aktiválódik-e**.
    """)

    st.subheader("🔧 Paraméterek")
    eta = st.slider("Tanulási ráta (η)", 0.01, 1.0, 0.1, step=0.01)
    epochs = st.slider("Epoch-ok száma", 1, 100, 20)

    X, T = generate_inputs()
    history = hebbian_learning(X, T, eta, epochs)

    # 📈 2D súlyváltozás
    st.subheader("📉 Súlyváltozások időben (2D)")
    fig, ax = plt.subplots()
    ax.plot(history[:, 0], label="w₀", linewidth=2)
    ax.plot(history[:, 1], label="w₁", linewidth=2)
    ax.set_xlabel("Iteráció")
    ax.set_ylabel("Súly érték")
    ax.set_title("Hebbian tanulás súlydinamikája")
    ax.legend()
    st.pyplot(fig)

    # 🌐 3D vizualizáció
    st.subheader("🌐 Súlypálya 3D térben")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=history[:, 0],
        y=history[:, 1],
        z=np.arange(len(history)),
        mode='lines+markers',
        marker=dict(size=4, color=np.arange(len(history)), colorscale='Viridis'),
        line=dict(width=3, color='darkblue')
    )])
    fig3d.update_layout(scene=dict(
        xaxis_title="w₀",
        yaxis_title="w₁",
        zaxis_title="Iteráció"
    ), margin=dict(l=0, r=0, t=30, b=0), height=500)
    st.plotly_chart(fig3d, use_container_width=True)

    # 📥 CSV export
    st.subheader("💾 Eredmények exportálása")
    df = pd.DataFrame(history, columns=["w₀", "w₁"])
    csv = df.to_csv(index_label="iteráció").encode("utf-8")
    st.download_button("⬇️ Súlyok letöltése CSV-ben", data=csv, file_name="hebb_weights.csv")

    # 📚 Tudományos háttér
    st.markdown("### 📚 Tudományos háttér")
    st.markdown(r"""
A **Hebbian tanulás** az egyik legegyszerűbb tanulási szabály,  
amely a biológiai neuronhálók **szinaptikus erősödését** modellezi.

#### 🧠 Alapelv:
> *„Azok a neuronok, amelyek együtt tüzelnek, együtt huzalozódnak.”*

#### 📐 Súlyfrissítési képlet:

$$
w_i \leftarrow w_i + \eta \cdot x_i \cdot t
$$

**Ahol:**

- \( w_i \): az *i*-edik bemeneti súly  
- \( \eta \): tanulási ráta  
- \( x_i \): a bemeneti neuron aktivációja  
- \( t \): a kimeneti neuron aktivációja (vagy célérték)

Ez a szabály akkor módosítja a súlyokat, ha a bemenet és kimenet **együtt aktiválódik**, vagyis korrelálnak.  
A Hebbian-elv alapvető szerepet játszik a **nem felügyelt tanulás** modellezésében, és megalapozza az asszociatív memóriák működését.

#### 📌 Alkalmazás:
- Biológiai tanulási mechanizmusok szimulációja  
- Nem felügyelt neurális modellek alapja  
- Szinaptikus erősségek időbeli változásának megértése
""")

# ✅ ReflectAI kompatibilitás
app = run
