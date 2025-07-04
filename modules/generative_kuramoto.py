import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

# 🌐 Hálózat generálása
def generate_graph(n_nodes, p):
    return nx.erdos_renyi_graph(n_nodes, p)

# 🔁 Kuramoto-szimuláció
def simulate_kuramoto(G, K, t_max=10, dt=0.05):
    N = len(G.nodes)
    A = nx.to_numpy_array(G)
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(1.0, 0.1, N)
    t = np.arange(0, t_max, dt)
    sync = []

    for _ in t:
        dtheta = omega + K / N * np.sum(A * np.sin(theta[:, None] - theta), axis=1)
        theta += dtheta * dt
        order_param = np.abs(np.sum(np.exp(1j * theta)) / N)
        sync.append(order_param)

    return t, sync, theta

# 📊 2D szinkronizációs plot
def plot_sync(t, sync):
    fig, ax = plt.subplots()
    ax.plot(t, sync)
    ax.set_xlabel("Idő")
    ax.set_ylabel("Szinkronizáció (R)")
    ax.set_title("Kuramoto szinkronizációs dinamika")
    st.pyplot(fig)

# 🌐 3D Plotly gráf
def plot_graph_3d(G, theta, palette):
    pos = nx.spring_layout(G, dim=3, seed=42)
    node_xyz = np.array([pos[v] for v in G.nodes()])
    edge_xyz = []

    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_xyz.append(((x0, x1), (y0, y1), (z0, z1)))

    edge_x, edge_y, edge_z = [], [], []
    for ex, ey, ez in edge_xyz:
        edge_x += list(ex) + [None]
        edge_y += list(ey) + [None]
        edge_z += list(ez) + [None]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='lightgray', width=1),
        opacity=0.5,
        name="Kapcsolatok"
    ))

    fig.add_trace(go.Scatter3d(
        x=node_xyz[:, 0], y=node_xyz[:, 1], z=node_xyz[:, 2],
        mode='markers',
        marker=dict(
            size=7,
            color=theta,
            colorscale=palette,
            line=dict(color='black', width=0.5)
        ),
        name="Oszcillátorok"
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# 🚀 Streamlit app
def run():
    st.set_page_config(layout="wide")
    st.title("🎲 Generatív Kuramoto hálózat")

    st.markdown("""
    Ez a modul egy **véletlenszerű gráfot** generál (Erdős–Rényi modell alapján), majd szimulálja rajta a **Kuramoto-modellt**, 
    amely az oszcillátorok fázisának szinkronizációját vizsgálja.
    """)

    # 🎛️ Paraméterek
    st.sidebar.header("⚙️ Beállítások")
    n_nodes = st.sidebar.slider("🧩 Csomópontok száma", 5, 100, 20)
    p = st.sidebar.slider("🔗 Él valószínűsége (p)", 0.0, 1.0, 0.1, 0.01)
    K = st.sidebar.slider("📡 Kapcsolási erősség (K)", 0.0, 10.0, 2.0, 0.1)
    t_max = st.sidebar.slider("⏱️ Szimuláció hossza (t_max)", 1, 50, 10)
    palette = st.sidebar.selectbox("🎨 Színséma", ["Viridis", "Turbo", "Electric", "Plasma", "Rainbow"])

    if st.button("▶️ Szimuláció indítása"):
        G = generate_graph(n_nodes, p)
        t, sync, theta = simulate_kuramoto(G, K, t_max)

        # 📈 2D diagram
        st.subheader("📈 Szinkronizációs dinamika")
        plot_sync(t, sync)

        # 🌐 3D gráf
        st.subheader("🌐 3D gráf vizualizáció – végső oszcillátorállapotokkal")
        plot_graph_3d(G, theta, palette)

        # 💾 CSV export
        st.subheader("💾 Szinkronizációs adatok letöltése")
        df_export = pd.DataFrame({
            "Idő (t)": t,
            "Szinkronizáció (R)": sync
        })
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ CSV letöltése",
            data=csv,
            file_name="generative_kuramoto.csv",
            mime="text/csv"
        )

        # 📘 Tudományos háttér
        st.markdown("### 📘 Tudományos háttér")

        st.latex(r"""
        \frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)
        """)

        st.markdown("""
        - **$\theta_i$**: az *i*-edik oszcillátor fázisa  
        - **$\omega_i$**: sajátfrekvencia  
        - **$A_{ij}$**: a gráf szomszédsági mátrixának elemei  
        - **$K$**: globális kapcsolódási erősség  
        - A rendszer szinkronizációs szintjét az ún. **order parameter** ($R$) mutatja:
        """)

        st.latex(r"""
        R(t) = \left| \frac{1}{N} \sum_{j=1}^N e^{i\theta_j(t)} \right|
        """)

        st.markdown("""
        - $R = 1$: teljes szinkron  
        - $R \approx 0$: dekoherens állapot  
        - A gráf struktúrája, az $N$, $p$ és $K$ értékek erősen befolyásolják a szinkronizáció mértékét.
        """)

        # 📝 Jegyzetek
        st.subheader("📝 Megfigyelések")
        st.text_area("Mit tapasztaltál a generált hálón?", placeholder="Írd ide...")

# ✅ ReflectAI-kompatibilitás
app = run
