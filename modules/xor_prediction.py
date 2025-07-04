import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import plotly.graph_objects as go


def run():
    st.set_page_config(layout="wide")
    st.title("🔀 XOR Predikció – Többrétegű Perceptron")

    # 🧭 Bevezetés
    st.markdown("""
    A klasszikus **XOR logikai feladat** nem oldható meg egyrétegű perceptronnal,  
    viszont egy **rejtett réteggel** ellátott MLP képes megtanulni.  
    Ez a modul vizualizálja a tanulási folyamatot, a döntési felületet, és a súlystruktúrát.
    """)

    # 🎛️ Paraméterek
    st.sidebar.header("🎚️ Paraméterek")
    hidden_size = st.sidebar.slider("Rejtett réteg mérete", 2, 10, 4)
    learning_rate = st.sidebar.slider("Tanulási ráta", 0.001, 0.1, 0.01, step=0.001)
    max_iter = st.sidebar.slider("Max iteráció", 100, 2000, 500, step=100)
    solver = st.sidebar.selectbox("Solver", ["adam", "sgd", "lbfgs"])
    activation = st.sidebar.selectbox("Aktivációs függvény", ["relu", "logistic", "tanh"])
    alpha = st.sidebar.slider("Regulárizációs erő (alpha)", 0.0001, 0.1, 0.001, step=0.0001)

    # 🧱 XOR adat
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # 🧠 Háló létrehozás + tanítás
    model = MLPClassifier(hidden_layer_sizes=(hidden_size,),
                          learning_rate_init=learning_rate,
                          max_iter=max_iter,
                          solver=solver,
                          activation=activation,
                          alpha=alpha,
                          random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    loss = log_loss(y, proba)

    # 📉 2D döntési függvény
    st.subheader("📈 Döntési felület (2D)")
    xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    plt.figure(figsize=(4, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k", s=100)
    plt.title(f"Pontosság: {acc * 100:.1f}%, Veszteség: {loss:.4f}")
    st.pyplot(plt.gcf())

    # 🌐 3D aktiválások
    st.subheader("🌐 Rejtett réteg aktiváció (3D)")
    act = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    fig3d = go.Figure(data=[go.Surface(z=act, x=xx, y=yy, colorscale="Viridis")])
    fig3d.update_layout(
        scene=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="P(rejtett=1)"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=600)
    st.plotly_chart(fig3d, use_container_width=True)

    # 🧩 Eredmény
    st.subheader("🎯 Eredmények")
    st.markdown(f"""
    - Háló struktúrája: **Input–{hidden_size}–Output**  
    - Aktiváció: **{activation}**  
    - Solver: **{solver}**  
    - Tanulási ráta: **{learning_rate}**  
    - Regulárizáció (alpha): **{alpha}**  
    - Iteráció: **{model.n_iter_} / {max_iter}**  
    - Pontosság: **{acc * 100:.2f}%**  
    - Log-loss: **{loss:.5f}**
    """)

    # 📁 CSV export
    st.subheader("💾 Súlyok exportálása CSV-ben")
    weight_vectors = np.concatenate([w.flatten() for w in model.coefs_])
    df = pd.DataFrame([weight_vectors], columns=[f"w{i}" for i in range(len(weight_vectors))])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Súlyok letöltése", data=csv, file_name="xor_weights.csv")

    # 📘 Tudományos háttér – Latex
    st.markdown("### 📙 Tudományos háttér")
    st.latex(r"""
    \text{XOR kimenet:}\quad
    y = x_1 \oplus x_2 = x_1(1 - x_2) + x_2(1 - x_1)
    """)
    st.latex(r"""
    \text{MLP kimenet:}\quad
    \hat y = \sigma^{(2)}\left(W^{(2)} \cdot \sigma^{(1)}(W^{(1)}x + b^{(1)}) + b^{(2)}\right)
    """)
    st.latex(r"""
    \text{Célfüggvény:}\quad
    \mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \hat y_i + (1 - y_i) \log(1 - \hat y_i) \right]
    """)
    st.markdown("""
    A **többrétegű perceptron** (MLP) képes a nemlineáris elválasztási problémák – mint az XOR – megoldására.  
    A háló megtanulja leképezni a nemlineáris döntési határt, a **log-loss** pedig a predikciók biztonságát méri.

    A paraméterek változtatásával tanulmányozhatjuk:
    - a **konvergencia** sebességét
    - a **túlillesztés** kockázatát (alacsony alpha értéknél)
    - a különböző aktivációs függvények viselkedését

    Ez a modul lehetőséget ad mélyebb **tanulásdinamika és hálóarchitektúra** vizsgálatra is.
    """)


# ReflectAI kompatibilitás
app = run
