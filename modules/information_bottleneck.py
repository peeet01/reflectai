import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# 🔧 Mutual information becslés (javított)
def compute_mutual_info(x, y):
    x_binned = pd.qcut(x, q=10, duplicates='drop').cat.codes
    return mutual_info_score(x_binned, y)

# 🎯 Information Bottleneck veszteségfüggvény
def information_bottleneck_loss(I_xt, I_ty, beta):
    return I_xt - beta * I_ty

# 🔁 IB-szimuláció rejtett zajjal
def simulate_ib(X, Y, beta, epochs=10, latent_dim=2):
    I_xt_list = []
    I_ty_list = []
    latent_history = []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for epoch in range(epochs):
        noise = np.random.normal(0, 1, X_scaled.shape)
        Z = X_scaled + beta * noise
        model = LogisticRegression(max_iter=1000)
        model.fit(Z, Y)
        preds = model.predict(Z)

        I_xt = compute_mutual_info(Z[:, 0], X_scaled[:, 0])
        I_ty = compute_mutual_info(Z[:, 0], preds)

        I_xt_list.append(I_xt)
        I_ty_list.append(I_ty)
        latent_history.append(Z)

    return I_xt_list, I_ty_list, latent_history

# 📈 3D vizualizáció (t-SNE)
def visualize_3d(latent_data, Y, step):
    tsne = TSNE(n_components=3, perplexity=30, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(latent_data[step])
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color=Y, colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(scene=dict(
        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3"
    ), margin=dict(l=0, r=0, t=30, b=0))
    return fig

# 🚀 Streamlit modul
def run():
    st.title("🔐 Information Bottleneck – Információs reprezentációk tömörítése")

    # 📘 Bevezető
    st.markdown("""
    A *Information Bottleneck* (IB) elmélet célja, hogy egy bemeneti változót **minimális információveszteséggel tömörítsünk**,  
    miközben megőrizzük a lehető legtöbb információt a **kimeneti célváltozóról**.  
    A megközelítés egyszerre szolgál **adatkompressziós** és **prediktív tanulási** célokat.

    Ez a modul interaktív módon vizualizálja, hogyan változik a rejtett reprezentáció **informatív és kompakt formája**,  
    ahogy a **kompresszió–pontosság arányát szabályozó β** paraméter változik.
    """)

    # ⚙️ Paraméterek
    st.sidebar.header("⚙️ Paraméterek")
    beta = st.sidebar.slider("Kompresszió–relevancia súly (β)", 0.01, 5.0, 1.0, step=0.05)
    latent_dim = st.sidebar.slider("Rejtett dimenzió", 2, 10, 3)
    steps = st.sidebar.slider("Epoch-ok száma", 1, 20, 10)
    step_idx = st.sidebar.slider("Vizualizált lépés", 0, steps - 1, 0)

    # 📊 Adatgenerálás (szintetikus osztályozási feladat)
    from sklearn.datasets import make_classification
    X, Y = make_classification(n_samples=300, n_features=5, n_informative=3, n_classes=3, random_state=42)

    I_xt_list, I_ty_list, latent_history = simulate_ib(X, Y, beta, steps, latent_dim)

    # 📈 Mutual Information grafikon
    st.subheader("📈 Információs mennyiségek alakulása")
    df_info = pd.DataFrame({
        "Epoch": np.arange(steps),
        "I(X;T)": I_xt_list,
        "I(T;Y)": I_ty_list
    })
    st.line_chart(df_info.set_index("Epoch"))

    # 🌐 t-SNE 3D vizualizáció
    st.subheader("🌐 Rejtett reprezentáció – 3D t-SNE")
    fig = visualize_3d(latent_history, Y, step_idx)
    st.plotly_chart(fig, use_container_width=True)

    # 💾 CSV export
    st.subheader("💾 Latens reprezentáció export")
    df_latent = pd.DataFrame(latent_history[step_idx])
    df_latent["label"] = Y
    st.dataframe(df_latent)
    csv = df_latent.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV letöltése", csv, "information_bottleneck_latent.csv", "text/csv")

    # 🧠 Tudományos háttér
    st.markdown("### 🧠 Tudományos háttér")
    st.latex(r"""
    \mathcal{L}_{IB} = I(X;T) - \beta \cdot I(T;Y)
    """)
    st.markdown("""
    - **\(X\)**: bemenet  
    - **\(T\)**: tömörített (rejtett) reprezentáció  
    - **\(Y\)**: célváltozó  
    - **\(I(A;B)\)**: mutual information A és B között  
    - **β**: szabályozza az információmegőrzés és tömörítés arányát

    A cél: megtalálni olyan \(T\)-t, amely **minél többet megtart \(Y\)-ról**,  
    miközben **minél kevesebb információt tartalmaz \(X\)-ből** – azaz csak a lényeges jellemzőket.

    **Kapcsolódó területek:** representation learning, variational inference, deep IB, unsupervised pretraining, AI fairness.
    """)

# ✅ ReflectAI kompatibilitás
app = run
