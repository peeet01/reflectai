import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def generate_data(N, timesteps, eta, init_std=0.1):
    x = np.random.randn(N, timesteps)
    W = np.random.randn(N, N) * init_std
    history = []

    for t in range(1, timesteps):
        y = np.tanh(W @ x[:, t-1])
        W += eta * np.outer(y, x[:, t-1])
        history.append(W.copy())

    return np.array(history), x

def visualize_weights_3d(data, method, step):
    matrix = data[step]  # shape: (N, N)

    if method == "Raw Grid":
        size = matrix.shape[0]
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        z = matrix
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
        return fig

    flattened = matrix  # shape: (N, N), each row = one sample
    try:
        if method == "PCA":
            model = PCA(n_components=3)
        elif method == "t-SNE":
            model = TSNE(n_components=3, perplexity=5, learning_rate='auto', init='pca')

        coords = model.fit_transform(StandardScaler().fit_transform(flattened))
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, color=z, colorscale='Plasma', opacity=0.8)
        )])
        return fig
    except Exception as e:
        st.error(f"Hiba a 3D vizualizáció során: {e}")
        return go.Figure()

def app():
    st.title("🧠 Critical Hebbian – Szinaptikus tanulás és komplexitás")

    # ▶️ Bevezetés és matematikai háttér
    st.markdown(r"""
    ### 🎯 Célkitűzés

    Ez a modul a **Hebbian tanulás** dinamikáját és annak **kritikus viselkedéshez** való viszonyát vizsgálja.  
    A Hebbian szabály egy pozitív visszacsatoláson alapuló tanulási mechanizmus, amely önmagában képes kritikus komplexitású mintázatok kialakítására.

    ### ⚙️ Matematikai háttér

    Hebbian tanulási szabály:
    $$
    \Delta w_{ij} = \eta \cdot y_i \cdot x_j
    $$

    A súlymátrix időfejlődése:
    $$
    W(t+1) = W(t) + \eta \cdot y(t) \cdot x(t)^T
    $$

    Ahol:
    - $x_j$ a bemeneti neuron aktivitása
    - $y_i = \tanh(Wx)$ a kimeneti neuron válasza
    - $W$ a szinaptikus súlymátrix
    - $\eta$ a tanulási ráta
    """)

    # 🎛️ Oldalsáv: Paraméterek
    st.sidebar.header("🔧 Paraméterek")
    N = st.sidebar.slider("Neuronok száma", 5, 100, 20, step=5)
    timesteps = st.sidebar.slider("Időlépések száma", 50, 500, 100, step=50)
    eta = st.sidebar.slider("Tanulási ráta (η)", 0.001, 0.1, 0.01)
    viz_type = st.sidebar.selectbox("Vizualizáció típusa", ["Raw Grid", "PCA", "t-SNE"])
    step = st.sidebar.slider("Lépés megjelenítése", 0, timesteps - 2, timesteps // 2)

    if st.sidebar.button("🔁 Szimuláció újraindítása"):
        st.session_state['data'], st.session_state['activity'] = generate_data(N, timesteps, eta)

    if 'data' not in st.session_state:
        st.session_state['data'], st.session_state['activity'] = generate_data(N, timesteps, eta)

    # 📊 Vizualizáció
    st.subheader("🎥 Súlymátrix vizualizáció 3D-ben")
    fig = visualize_weights_3d(st.session_state['data'], viz_type, step)
    st.plotly_chart(fig, use_container_width=True)

    # 💾 CSV mentés
    st.subheader("💾 Súlymátrix mentése CSV-be")
    W_df = pd.DataFrame(st.session_state['data'][step])
    st.dataframe(W_df)
    csv = W_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Letöltés CSV formátumban", csv, "hebbian_weights.csv", "text/csv")

    # 📖 Tudományos leírás
    st.markdown("## 📚 Tudományos magyarázat és következtetések")
    st.markdown(r"""
    A Hebbian tanulás egy **alulról szerveződő**, lokális szabályon alapuló mechanizmus.  
    Az alábbi jelenségek figyelhetők meg:

    ### 🔍 Megfigyelések:
    - A tanulási ráta ($\eta$) értékétől függően a hálózat **konvergál**, **divergál**, vagy **önszerveződő mintázatokat** mutat
    - A PCA és t-SNE módszerek **nemtriviális struktúrákat** fednek fel a súlytérben
    - A rendszer bizonyos paramétereknél **kritikus pont** közelébe kerülhet

    ### 💡 Tudományos jelentőség:
    A kritikus Hebbian tanulás kapcsolódik a **kritikusság elméletéhez**, mely szerint egy rendszer legnagyobb komplexitását és alkalmazkodóképességét a **fázishatár körül** éri el.  
    Ez az állapot jellemző a biológiai idegrendszerekre is, ahol a **plaszticitás** és **emergencia** kulcsfontosságú szerepet játszanak.
    """)

# 🔁 Kötelező belépési pont
app = app
