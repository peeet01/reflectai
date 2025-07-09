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
    flattened = data[step].reshape(-1, 1)

    if method == "PCA":
        model = PCA(n_components=3)
    elif method == "t-SNE":
        model = TSNE(n_components=3, perplexity=10, learning_rate='auto', init='pca')
    else:  # Raw Grid
        size = int(np.sqrt(flattened.shape[0]))
        coords = np.indices((size, size)).reshape(2, -1).T
        x, y = coords[:, 0], coords[:, 1]
        z = flattened.flatten()
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=4, color=z, colorscale='Viridis', opacity=0.8)
        )])
        return fig

    coords = model.fit_transform(StandardScaler().fit_transform(flattened))
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, color=z, colorscale='Plasma', opacity=0.8)
    )])
    return fig

def run():
    st.title("🌋 Critical Hebbian – Szinaptikus tanulás és komplexitás")

    st.markdown("""
    ### 🧠 Tudományos célkitűzés
    Ez a modul a **Hebbian tanulás** dinamikáját és annak **kritikus viselkedéshez** való viszonyát mutatja be.  
    A Hebbian-szabály önmagában pozitív visszacsatolású, és bizonyos feltételek mellett **kritikus fázishatár** közelébe tolhatja a hálózatot.

    ### ⚙️ Matematikai háttér

    Hebbian tanulás:

    $$
    \Delta w_{ij} = \eta \cdot y_i \cdot x_j
    $$

    Ahol:
    - $x_j$: bemeneti neuron aktivitása
    - $y_i$: kimeneti neuron aktivitása (pl. $\\tanh(Wx)$)
    - $w_{ij}$: szinaptikus súly
    - $\\eta$: tanulási ráta

    Időben változó súlymátrix:  
    $$
    W(t+1) = W(t) + \eta \cdot y(t) \cdot x(t)^T
    $$

    ### 🎯 Cél
    - Szemléltetni a Hebbian tanulás hatását a súlytér szerkezetére
    - Feltárni a tanulási dinamika esetleges **kritikus pontközeli** viselkedését
    - 3D-s vizualizációval értelmezni a súlytér alakulását különböző nézőpontokból

    ---
    """)

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

    st.subheader("🎥 Súlymátrix vizualizáció 3D-ben")
    fig = visualize_weights_3d(st.session_state['data'], viz_type, step)
    st.plotly_chart(fig, use_container_width=True)

    # Export
    st.subheader("💾 Súlymátrix mentése CSV-be")
    W_df = pd.DataFrame(st.session_state['data'][step])
    st.dataframe(W_df)
    csv = W_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Letöltés CSV formátumban", csv, "hebbian_weights.csv", "text/csv")

    with st.expander("📚 Tudományos magyarázat és következtetések"):
        st.markdown(r"""
        A Hebbian szabály előnye, hogy egyszerű és biológiailag motivált, azonban **önmagában instabil** lehet.  
        Képes **kritikus viselkedés** közelébe tolni a hálózatot, ami maximális komplexitást és információfeldolgozási képességet biztosít.

        ### 🔍 Megfigyelhető jelenségek:
        - A súlytér szerkezete **nemlineárisan változik** az idő során
        - A PCA/t-SNE vizualizációk révén **rejtett minták** és **topológiai szerveződések** figyelhetők meg
        - A tanulási ráta befolyásolja, hogy a hálózat **konvergál-e**, **divergál-e**, vagy **önszerveződő mintázatokat** mutat

        ### 💡 Tudományos jelentőség:
        A Hebbian tanulás egyszerű szabálya mögött egy rendkívül gazdag dinamika rejlik, amely kapcsolatban áll a **kritikusság**, **emergencia** és **önszerveződés** fogalmaival.  
        A megfelelő tanulási paraméterek mellett a hálózat a legnagyobb **adaptivitási** és **plaszticitási** potenciált mutatja.
        """)

# Kötelező belépési pont
app = run
