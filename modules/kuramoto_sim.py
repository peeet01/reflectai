import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go

# 🔁 Kuramoto lépés
def kuramoto_step(theta, K, A, omega, dt):
    N = len(theta)
    dtheta = omega + (K / N) * np.sum(A * np.sin(theta[:, None] - theta), axis=1)
    return theta + dt * dtheta

# 🔢 Szinkronizációs index
def compute_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))

# 🌐 Hálózat generálás
def generate_graph(N, graph_type):
    if graph_type == "Teljes":
        return nx.complete_graph(N)
    elif graph_type == "Véletlen (Erdős-Rényi)":
        return nx.erdos_renyi_graph(N, p=0.3)
    elif graph_type == "Kis világ (Watts-Strogatz)":
        return nx.watts_strogatz_graph(N, k=4, p=0.3)
    elif graph_type == "Skálafüggetlen (Barabási-Albert)":
        return nx.barabasi_albert_graph(N, m=2)
    else:
        return nx.complete_graph(N)

# 🚀 Streamlit alkalmazás
def run():
    st.set_page_config(layout="wide")
    st.title("🧠 Kuramoto Szinkronizáció – Dinamikus Oszcillátor Modell")

    st.markdown("""
    A **Kuramoto-modell** egy nemlineáris dinamikai modell, amely **több oszcillátor fázisának** időbeli szinkronizációját írja le.
    A modell fontos jelenségeket magyaráz: biológiai ritmusok, idegi kapcsolatok, hálózati koordináció stb.
    """)

    # 🎛️ Beállítások
    st.sidebar.header("⚙️ Szimulációs paraméterek")
    N = st.sidebar.slider("Oszcillátorok száma (N)", 5, 100, 30)
    K = st.sidebar.slider("Kapcsolódási erősség (K)", 0.0, 10.0, 2.0, 0.1)
    steps = st.sidebar.slider("Iterációk száma", 100, 2000, 500, 100)
    graph_type = st.sidebar.selectbox("Hálózat típusa", [
        "Teljes", "Véletlen (Erdős-Rényi)", "Kis világ (Watts-Strogatz)", "Skálafüggetlen (Barabási-Albert)"
    ])
    palette = st.sidebar.selectbox("Színséma (3D)", ["Turbo", "Viridis", "Electric", "Hot", "Rainbow"])

    dt = 0.05  # időlépés

    # 🔁 Szimuláció
    st.subheader("🔄 Szimuláció futtatása")
    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)
    G = generate_graph(N, graph_type)
    A = nx.to_numpy_array(G)

    order_params = []
    for _ in range(steps):
        theta = kuramoto_step(theta, K, A, omega, dt)
        order_params.append(compute_order_parameter(theta))

    # 📊 2D szinkronizációs index
    st.subheader("📈 Szinkronizációs index (R)")
    st.line_chart(order_params)

    # 🌐 3D hálózat megjelenítés
    st.subheader("🌐 3D vizualizáció – Oszcillátor fázisok színkóddal")

    pos = nx.circular_layout(G, dim=3)
    node_x, node_y, node_z = zip(*[pos[n] for n in G.nodes()])
    edge_x, edge_y, edge_z = [], [], []

    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='lightgray', width=1),
        opacity=0.4,
        name='Kötések'
    ))

    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=7,
            color=theta,
            colorscale=palette,
            opacity=0.9,
            line=dict(color='black', width=0.5)
        ),
        name='Oszcillátorok'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False)
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # 📘 Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")

    st.latex(r"""
    \frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)
    """)

    st.markdown("""
    - **$\theta_i$**: az *i*-edik oszcillátor fázisa  
    - **$\omega_i$**: sajátfrekvencia (normál eloszlásból)  
    - **$K$**: globális kapcsolódási erősség  
    - **$A_{ij}$**: szomszédsági mátrix a hálózatra  
    - A szinkronizáció mértékét az **orderr paraméter ($R$)** mutatja:
    """)

    st.latex(r"""
    R(t) = \left| \frac{1}{N} \sum_{j=1}^N e^{i\theta_j(t)} \right|
    """)

    st.markdown("""
    - $R=1$ esetén tökéletes szinkron  
    - $R \approx 0$ esetén káoszos, dekoherens állapot  
    - A modell vizsgálható különféle hálózati topológiákon

    **Alkalmazás:**  
    - Neurális populációk szinkronizációja  
    - Kémiai oszcillátorok  
    - Hálózati dinamika, például tűzfal vagy robotrajok koordinációja
    """)

    # 💾 Szinkronizációs adatok exportálása (CSV)
    import pandas as pd

    df_export = pd.DataFrame({
        "Időlépés": np.arange(1, steps + 1),
        "Szinkronizáció (R)": order_params
    })

    csv = df_export.to_csv(index=False).encode("utf-8")

    st.subheader("💾 Szinkronizációs adatok letöltése")
    st.download_button(
        label="⬇️ CSV letöltése",
        data=csv,
        file_name="kuramoto_sync.csv",
        mime="text/csv"
    )

    # 🗒️ Megfigyelések
    st.subheader("📝 Megfigyelések és jegyzetek")
    st.text_area("Mit tapasztaltál a szinkronizáció során?", placeholder="Írd ide...")

# ✅ ReflectAI-kompatibilitás
app = run
