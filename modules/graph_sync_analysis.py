import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import plotly.graph_objects as go

# 🔄 Kuramoto lépés
def kuramoto_step(theta, omega, A, K, dt):
    N = len(theta)
    dtheta = omega + (K / N) * np.sum(A * np.sin(np.subtract.outer(theta, theta)), axis=1)
    return theta + dtheta * dt

# ▶️ Szimuláció
def run_simulation(G, steps, dt, K):
    N = len(G)
    theta = np.random.rand(N) * 2 * np.pi
    omega = np.random.randn(N)
    A = nx.to_numpy_array(G)

    r_values = []
    theta_history = []

    for _ in range(steps):
        theta = kuramoto_step(theta, omega, A, K, dt)
        r = np.abs(np.mean(np.exp(1j * theta)))
        r_values.append(r)
        theta_history.append(theta.copy())
    return r_values, theta_history

# 🖼️ Gráf rajzolás (2D)
def draw_graph(G, theta=None):
    pos = nx.spring_layout(G, seed=42)
    node_colors = 'lightblue'
    if theta is not None:
        norm_theta = (theta % (2 * np.pi)) / (2 * np.pi)
        node_colors = cm.hsv(norm_theta)

    fig, ax = plt.subplots()
    nx.draw(G, pos, node_color=node_colors, edge_color='gray', with_labels=True, ax=ax)
    ax.set_title("🧠 Gráfstruktúra (fázis színezéssel)")
    st.pyplot(fig)

# 🌐 3D gráf megjelenítés
def draw_graph_3d(G, theta=None):
    pos = nx.spring_layout(G, dim=3, seed=42)
    xyz = np.array([pos[v] for v in sorted(G)])

    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    color_list = "lightblue"
    if theta is not None:
        norm_theta = (theta % (2 * np.pi)) / (2 * np.pi)
        node_colors = cm.hsv(norm_theta)
        color_list = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, _ in node_colors]

    fig = go.Figure(data=[
        go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines',
                     line=dict(color='gray', width=2), hoverinfo='none'),
        go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                     mode='markers', marker=dict(size=6, color=color_list), hoverinfo='text')
    ])
    fig.update_layout(title="🌐 3D gráfvizualizáció", showlegend=False,
                      margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig)

# 🚀 Fő alkalmazás
def run():
    st.set_page_config(layout="wide")
    st.title("🔗 Gráfalapú szinkronizációs analízis")

    st.markdown("""
Ez a modul a **Kuramoto-modell** segítségével vizsgálja, hogyan szinkronizálódnak oszcillátorok különböző gráfhálózatokon.

Megérthetjük, hogy a gráf szerkezete hogyan befolyásolja a szinkronizáció gyorsaságát és mértékét.
""")

    # 🎛️ Beállítások
    st.sidebar.header("⚙️ Beállítások")
    graph_type = st.sidebar.selectbox("Gráftípus", ["Erdős–Rényi", "Kör", "Rács", "Teljes gráf"])
    N = st.sidebar.slider("Csomópontok száma", 5, 100, 30)
    K = st.sidebar.slider("Kapcsolódási erősség (K)", 0.0, 10.0, 2.0)
    steps = st.sidebar.slider("Szimulációs lépések", 10, 1000, 300)
    dt = st.sidebar.slider("Időlépés (dt)", 0.001, 0.1, 0.01)
    er_p = st.sidebar.slider("Erdős–Rényi élvalószínűség", 0.05, 1.0, 0.1, step=0.05)
    show_3d = st.sidebar.checkbox("🌐 3D gráf megjelenítés")

    # ▶️ Szimuláció indítása
    if st.button("▶️ Szimuláció indítása"):
        if graph_type == "Erdős–Rényi":
            G = nx.erdos_renyi_graph(N, er_p)
        elif graph_type == "Kör":
            G = nx.cycle_graph(N)
        elif graph_type == "Rács":
            side = int(np.sqrt(N))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
        elif graph_type == "Teljes gráf":
            G = nx.complete_graph(N)

        with st.spinner("⏳ Szimuláció folyamatban..."):
            start = time.time()
            r_values, theta_hist = run_simulation(G, steps, dt, K)
            end = time.time()

        st.success(f"✅ Szimuláció befejezve ({end - start:.2f} másodperc)")
        st.metric("📈 Végső szinkronizációs érték (r)", f"{r_values[-1]:.3f}")
        st.metric("📊 Átlagos szinkronizáció (⟨r⟩)", f"{np.mean(r_values):.3f}")

        st.subheader("📉 Szinkronizáció időbeli lefutása")
        fig1, ax1 = plt.subplots()
        ax1.plot(r_values)
        ax1.set_xlabel("Időlépések")
        ax1.set_ylabel("Szinkronizáció (r)")
        ax1.set_title("Kuramoto-szinkronizációs dinamika")
        st.pyplot(fig1)

        st.subheader("🧠 Kezdeti állapot")
        draw_graph_3d(G, theta_hist[0]) if show_3d else draw_graph(G, theta_hist[0])

        st.subheader("🎯 Szinkronizált állapot")
        draw_graph_3d(G, theta_hist[-1]) if show_3d else draw_graph(G, theta_hist[-1])

        st.subheader("📝 Jegyzet")
        notes = st.text_area("Megfigyelések, következtetések:", height=150)
        if notes:
            st.download_button("💾 Jegyzet mentése", data=notes, file_name="sync_notes.txt")

    # 📘 Tudományos háttér
    st.subheader("📘 Tudományos háttér")
    st.markdown(r"""
A **Kuramoto-modell** a szinkronizációs jelenségek klasszikus leírása, ahol oszcillátorok egy gráfhálózaton keresztül hatnak egymásra.

#### Fázisdinamika:
$$
\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_j A_{ij} \sin(\theta_j - \theta_i)
$$

Itt:
- $\theta_i$: az $i$-edik oszcillátor fázisa
- $\omega_i$: természetes frekvenciája
- $K$: kapcsolódási erősség
- $A_{ij}$: gráf szomszédsági mátrixa

#### Rendparaméter:
$$
r(t) = \left| \frac{1}{N} \sum_{j=1}^N e^{i\theta_j(t)} \right|
$$

Ez méri a szinkronizáció mértékét:
- $r = 1$: teljes szinkron
- $r \approx 0$: rendezetlenség

A gráf szerkezete jelentősen befolyásolja a szinkronizáció sebességét és végső állapotát.
""")

# ReflectAI kompatibilitás
app = run
