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
    Ez a modul vizualizálja a tanulási folyamatot, súlyokat, döntési felületet és az aktivációt.
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
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_size,),
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        solver=solver,
        random_state=42,
        verbose=False
    )
    model.fit(X, y)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]
    acc = accuracy_score(y, preds)

    # 📉 2D döntési felület
    st.subheader("📈 Döntési felület (2D)")
    xx, yy = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    plt.figure(figsize=(4,4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(X[:,0], X[:,1], c=y, cmap="RdBu", edgecolor="k", s=100)
    plt.title(f"Pontosság: {acc*100:.1f}%")
    st.pyplot(plt.gcf())

    # 🌐 3D aktiváció
    st.subheader("🌐 Rejtett réteg aktiváció (3D)")
    act = model.predict_proba(grid)[:,1].reshape(xx.shape)
    fig3d = go.Figure(data=[go.Surface(z=act, x=xx, y=yy, colorscale="Viridis")])
    fig3d.update_layout(
        scene=dict(xaxis_title="x₁", yaxis_title="x₂", zaxis_title="P(kimenet=1)"),
        margin=dict(l=10,r=10,t=50,b=10),
        height=600)
    st.plotly_chart(fig3d, use_container_width=True)

    # 📉 Veszteséggörbe (ha elérhető)
    if hasattr(model, "loss_curve_"):
        st.subheader("📉 Veszteséggörbe")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(model.loss_curve_, label="Log-loss")
        ax_loss.set_xlabel("Iteráció")
        ax_loss.set_ylabel("Veszteség")
        ax_loss.set_title("Tanulási görbe")
        ax_loss.grid(True)
        st.pyplot(fig_loss)

    # 📊 Súlymátrix vizualizáció
    st.subheader("🧮 Súlymátrix")
    all_weights = []
    for i, coef in enumerate(model.coefs_):
        flat = coef.flatten()
        for j, val in enumerate(flat):
            all_weights.append((f"W{i+1}_{j+1}", val))
    df_weights = pd.DataFrame(all_weights, columns=["Súly", "Érték"])
    st.dataframe(df_weights.set_index("Súly").T)

    # 🧩 Eredmény
    st.subheader("🎯 Eredmények")
    st.markdown(f"""
    - Háló struktúra: **Input–{hidden_size}–Output**  
    - Solver: **{solver}**  
    - Tanulási ráta: **{learning_rate}**  
    - Iteráció: **{model.n_iter_}** / {max_iter}  
    - Pontosság: **{acc*100:.2f}%**
    """)

    # 📁 CSV export
    st.subheader("💾 Súlyok exportálása CSV-ben")
    weights_only = np.hstack([coef.flatten() for coef in model.coefs_])
    df_csv = pd.DataFrame(weights_only.reshape(1, -1),
                          columns=[f"w{i}" for i in range(len(weights_only))])
    csv = df_csv.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Súlyok letöltése", data=csv, file_name="xor_weights.csv")

    # 📘 Tudományos háttér – Mélyített magyarázatokkal
    st.markdown("### 📙 Tudományos háttér")

    st.latex(r"""
    \text{XOR kimenet: } y = 
    \begin{cases}
        1 & \text{ha } x_1 \ne x_2 \\
        0 & \text{ha } x_1 = x_2
    \end{cases}
    """)

    st.latex(r"""
    \text{MLP kimenet: } 
    \hat{y} = \sigma\left(W^{(2)} \cdot \sigma(W^{(1)}x + b^{(1)}) + b^{(2)}\right)
    """)

    st.latex(r"""
    \text{Aktiváció (ReLU vagy sigmoid): } 
    \sigma(x) = \frac{1}{1 + e^{-x}} \quad \text{vagy} \quad \max(0, x)
    """)

    st.latex(r"""
    \text{Log-veszteség: } 
    L = -\frac{1}{N}\sum_i\left[y_i\log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)\right]
    """)

    st.markdown("""
    #### 🔬 Magyarázat

    - A **log-loss** minimalizálása során a modell egyre pontosabban becsüli a valószínűségeket.
    - Az **egyrétegű perceptron** nem képes megtanulni az XOR nemlineáris elválasztását.
    - Egy **rejtett réteg** viszont lehetővé teszi a nemlineáris döntési határ megtanulását.
    - A **súlyok** elemzésével megfigyelhető, hogyan reprezentálja a modell a logikai kapcsolatokat.

    #### 📌 Következtetés

    Ez a modul vizuálisan és matematikailag is demonstrálja, hogyan képes egy egyszerű MLP megoldani a híres **XOR problémát**.
    """)

# ReflectAI kompatibilitás
app = run
