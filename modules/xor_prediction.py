import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

def run():
    st.title("🔁 XOR Prediction – Tudományos Neurális Hálózat Playground")

    st.markdown("""
    Ez a modul egy neurális hálózatot alkalmaz az **XOR logikai művelet** megtanulására,  
    tudományosan validálható módon: aktivációk, solverek, architektúra és loss analízis mellett.
    """)

    # --- Adat generálás ---
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # --- Felhasználói beállítások ---
    hidden_layer_size = st.slider("🔧 Rejtett réteg mérete", 2, 20, 4)
    max_iter = st.slider("🔁 Iterációk száma", 100, 2000, 500, step=100)
    activation = st.selectbox("📊 Aktivációs függvény", ["tanh", "relu", "logistic"])
    solver = st.selectbox("🧲 Solver algoritmus", ["adam", "sgd", "lbfgs"])
    learning_rate_init = st.slider("📈 Tanulási ráta", 0.001, 1.0, 0.01, step=0.01)
    show_3d = st.checkbox("🌐 3D vizualizáció", value=True)

    # --- Modell betanítása ---
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    # --- Kiértékelés ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"🌟 Modell pontossága: {acc:.2f} | Tanítás ideje: {end-start:.3f} másodperc")

    # --- Loss görbe ---
    if hasattr(model, 'loss_curve_'):
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(model.loss_curve_)
        ax_loss.set_xlabel("Iteráció")
        ax_loss.set_ylabel("Veszteség")
        ax_loss.set_title("📉 Tanulási veszteség")
        st.pyplot(fig_loss)

    # --- Konfúziós mátrix ---
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
    st.pyplot(fig_cm)

    # --- 3D döntési felület ---
    if show_3d:
        xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        zz = np.array([
            model.predict([[x, y]])[0] for x, y in zip(np.ravel(xx), np.ravel(yy))
        ]).reshape(xx.shape)

        fig = go.Figure(data=[
            go.Surface(z=zz, x=xx, y=yy, colorscale='Viridis', opacity=0.85)
        ])
        fig.update_layout(
            title="🧠 Döntési felület – tanult reprezentáció",
            scene=dict(xaxis_title='X₁', yaxis_title='X₂', zaxis_title='Predikció'),
            margin=dict(l=0, r=0, t=60, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Rejtett réteg aktivációk ---
    with st.expander("🧩 Rejtett réteg neuronjainak aktivációi"):
        if activation == "logistic":
            act_fn = lambda x: 1 / (1 + np.exp(-x))
        elif activation == "tanh":
            act_fn = np.tanh
        elif activation == "relu":
            act_fn = lambda x: np.maximum(0, x)
        else:
            act_fn = lambda x: x

        W1 = model.coefs_[0]
        b1 = model.intercepts_[0]
        hidden_outputs = act_fn(np.dot(X, W1) + b1)

        fig_hidden, ax_hidden = plt.subplots()
        im = ax_hidden.imshow(hidden_outputs, cmap='coolwarm', aspect='auto')
        ax_hidden.set_xticks(range(hidden_outputs.shape[1]))
        ax_hidden.set_yticks(range(X.shape[0]))
        ax_hidden.set_xlabel("Neuron index")
        ax_hidden.set_ylabel("Bemeneti minta index")
        ax_hidden.set_title("🔍 Aktivációk a rejtett rétegben")
        fig_hidden.colorbar(im)
        st.pyplot(fig_hidden)

    # --- Tudományos magyarázat ---
    with st.expander("📘 Matematikai háttér"):
        st.markdown(r"""
        Az **XOR** (exclusive OR) probléma egy nemlineárisan szeparálható logikai művelet:

        $$
        \text{XOR}(x_1, x_2) =
        \begin{cases}
        0, & \text{ha } x_1 = x_2 \\
        1, & \text{ha } x_1 \neq x_2
        \end{cases}
        $$

        Egyetlen rétegű perceptron nem képes ezt megtanulni, mert a bemeneti tér nem szeparálható egyetlen egyenes mentén.

        A tanulási modell egy **többrétegű perceptron (MLP)**:
        $$
        \hat{y} = \sigma \left( W^{(2)} \cdot \sigma(W^{(1)}x + b^{(1)}) + b^{(2)} \right)
        $$

        Itt:
        - $\sigma$ az aktivációs függvény (pl. $\tanh$, $\text{ReLU}$)
        - $W^{(1)}, W^{(2)}$ a súlymátrixok
        - $b^{(1)}, b^{(2)}$ a bias vektorok

        A célfüggvény (pl. MSE):
        $$
        \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \left(y_i - \hat{y}_i\right)^2
        $$

        A tanulás célja: a veszteség minimalizálása a teljes tanítókészleten.
        """)

# Kötelező ReflectAI belépési pont
app = run
