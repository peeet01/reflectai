import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

def run():
    st.set_page_config(layout="wide")
    st.title("🔀 XOR Predikció – Többrétegű Perceptron")

    # 🧭 Bevezetés
    st.markdown("""
    A klasszikus **XOR logikai feladat** nem oldható meg egyrétegű perceptronnal,  
    viszont egy **rejtett réteggel** ellátott MLP képes megtanulni.  
    A modul bemutatja, hogyan változnak a súlyok, a döntési felület, és milyen pontossággal oldja meg a háló a problémát.
    """)

    # 🎛️ Paraméterek
    st.sidebar.header("🎚️ Paraméterek")
    hidden_size = st.sidebar.slider("Rejtett réteg mérete", 2, 10, 4)
    learning_rate = st.sidebar.slider("Tanulási ráta", 0.001, 0.1, 0.01, step=0.001)
    max_iter = st.sidebar.slider("Max iteráció", 100, 2000, 500, step=100)
    solver = st.sidebar.selectbox("Solver", ["adam", "sgd", "lbfgs"])

    # 🧱 XOR adat
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])

    # 🧠 Háló létrehozás + tanítás
    model = MLPClassifier(hidden_layer_sizes=(hidden_size,),
                          learning_rate_init=learning_rate,
                          max_iter=max_iter,
                          solver=solver,
                          random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    # 📉 2D döntési függvény
    st.subheader("📈 Döntési felület (2D)")
    xx, yy = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    plt.figure(figsize=(4,4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(X[:,0], X[:,1], c=y, cmap="RdBu", edgecolor="k", s=100)
    plt.title(f"Pontosság: {acc*100:.1f}%")
    st.pyplot(plt.gcf())

    # 🌐 3D aktiválások
    st.subheader("🌐 Rejtett réteg aktiváció (3D)")
    act = model.predict_proba(grid)[:,1].reshape(xx.shape)
    fig3d = go.Figure(data=[go.Surface(z=act, x=xx, y=yy, colorscale="Viridis")])
    fig3d.update_layout(
        scene=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="P(rejtett=1)"),
        margin=dict(l=10,r=10,t=50,b=10),
        height=600)
    st.plotly_chart(fig3d, use_container_width=True)

    # 🧩 Eredmény
    st.subheader("🎯 Eredmények")
    st.markdown(f"- Háló struktúrája: **Input–{hidden_size}–Output**\n"
                f"- Solver: **{solver}**\n"
                f"- Tanulási ráta: **{learning_rate}**\n"
                f"- Iteráció: **{model.n_iter_}** / {max_iter}\n"
                f"- Pontosság: **{acc*100:.2f}%**")

    # 📁 CSV export
    st.subheader("💾 Súlyok exportálása CSV-ben")
    weights = np.hstack([coef.flatten() for coef in model.coefs_])
    df = pd.DataFrame(weights.reshape(1, -1),
                      columns=[f"w{i}" for i in range(len(weights))])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Súlyok letöltése", data=csv, file_name="xor_weights.csv")

    # 📘 Tudományos háttér – Latex
    st.markdown("### 📙 Tudományos háttér")
    st.latex(r"""
    y = 
    \begin{cases}
        1 & \text{ha } x_1 \oplus x_2 = 1,\\
        0 & \text{különben}
    \end{cases}
    """)
    st.latex(r"""
    \text{MLP architektúra: } f(x) = \sigma\bigl(W^{(2)}\,\sigma(W^{(1)}x+b^{(1)}) + b^{(2)}\bigr)
    """)
    st.latex(r"""
    \text{Célfüggvény (log-loss): } 
    L = -\frac{1}{N}\sum_i\left[y_i\log \hat y_i + (1-y_i)\log(1-\hat y_i)\right]
    """)
    st.markdown("""
    A modellben:
    - \(W^{(1)}, b^{(1)}\): bemenet → rejtett réteg súlyai
    - \(W^{(2)}, b^{(2)}\): rejtett réteg → kimenet
    - \(\sigma\): nemlinearitás (ReLU vagy logistic)
    - A log-loss minimalizálásával a háló megtanulja megoldani a XOR problémát, amit egyrétegű perceptron nem tud.

    A pontosság mutatja, hogy minden bemeneti kombinációt helyesen prediktáltunk‑e.
    """)

# ReflectAI kompatibilitás
app = run
