import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def generate_input_data(kind, dim, samples):
    if kind == "Gauss":
        return np.random.randn(samples, dim)
    elif kind == "Uniform":
        return np.random.rand(samples, dim) - 0.5
    elif kind == "Clustered":
        centers = np.random.randn(3, dim) * 2
        data = np.vstack([np.random.randn(samples // 3, dim) + c for c in centers])
        return data
    else:
        return np.zeros((samples, dim))

def oja_learning(X, eta, epochs):
    # opcionális: középre tolás
    Xc = X - np.mean(X, axis=0, keepdims=True)

    # inicializálás
    w = np.random.randn(X.shape[1])
    w /= np.linalg.norm(w) + 1e-12

    history = []

    for _ in range(epochs):
        idx = np.random.permutation(len(Xc))
        for x in Xc[idx]:
            y = np.dot(w, x)
            dw = eta * y * (x - y * w)  # Oja update
            w += dw
            history.append(w.copy())

        # numerikai stabilizálás
        nrm = np.linalg.norm(w)
        if not np.isfinite(nrm) or nrm > 1e6:
            w /= nrm + 1e-12

    return np.array(history)

def run():
    st.title("🧠 Oja Learning – PCA-szerű tanulás")
    st.markdown("A modell egyetlen neuron súlyait tanítja meg úgy, hogy a bemeneti adatok fő komponensét tanulja meg.")

    kind = st.selectbox("Bemenet típusa", ["Gauss", "Uniform", "Clustered"])
    dim = st.slider("Dimenzió", 2, 5, 3)
    samples = st.slider("Minták száma", 100, 1000, 300, step=50)
    eta = st.slider("Tanulási ráta (η)", 0.001, 0.1, 0.01, step=0.001)
    epochs = st.slider("Epoch-ok száma", 1, 20, 10)

    X = generate_input_data(kind, dim, samples)
    history = oja_learning(X, eta, epochs)

    st.subheader("📈 Súlyfejlődés (2D)")
    fig, ax = plt.subplots()
    for i in range(history.shape[1]):
        ax.plot(history[:, i], label=f"w{i}")
    ax.set_xlabel("Iteráció")
    ax.set_ylabel("Súly érték")
    ax.set_title("Súlyváltozás Oja tanulás során")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Súlyvektor pálya 3D-ben")
    if history.shape[1] >= 3:
        fig3d = go.Figure(data=[go.Scatter3d(
            x=history[:, 0],
            y=history[:, 1],
            z=history[:, 2],
            mode="lines+markers",
            marker=dict(size=3),
            line=dict(width=2)
        )])
        fig3d.update_layout(
            scene=dict(xaxis_title="w0", yaxis_title="w1", zaxis_title="w2"),
            margin=dict(l=0, r=0, b=0, t=30),
            height=500
        )
        st.plotly_chart(fig3d)
    else:
        st.info("A 3D vizualizációhoz legalább 3 dimenzió szükséges.")

    st.subheader("📥 CSV Export")
    df = pd.DataFrame(history, columns=[f"w{i}" for i in range(history.shape[1])])
    csv = df.to_csv(index_label="iteráció").encode("utf-8")
    st.download_button("Súlytörténet letöltése", data=csv, file_name="oja_weights.csv")

    st.markdown("### 📚 Tudományos háttér")
    st.markdown(r"""
Az **Oja-tanulási szabály** a Hebbian tanulás normalizált változata, amely stabilizálja a súlyok hosszát.

**Cél:** a neuron megtanulja a bemenet fő irányát – hasonlóan a főkomponens-analízishez (PCA).

**Tanulási szabály:**

$$
\Delta w = \eta \cdot y \cdot (x - y \cdot w)
$$

Ahol:
- \( x \) a bemenet,
- \( w \) a súlyvektor,
- \( y = w^T x \) a neuron kimenete,
- \( \eta \) a tanulási ráta.

A súlyok konvergálnak az első főkomponens irányába.

**Felhasználás:**
- Neurális PCA tanítás
- Érzékenységi irányok tanulása
- Nem felügyelt tanulás alapmodellje
""")

app = run
