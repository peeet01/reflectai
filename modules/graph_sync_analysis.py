import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import time

def kuramoto_step(theta, omega, A, K, dt):
    N = len(theta)
    dtheta = omega + (K / N) * np.sum(A * np.sin(np.subtract.outer(theta, theta)), axis=1)
    return theta + dtheta * dt

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

def draw_graph(G, theta=None):
    pos = nx.spring_layout(G, seed=42)
    if theta is not None:
        norm_theta = (theta % (2 * np.pi)) / (2 * np.pi)
        node_colors = cm.hsv(norm_theta)
    else:
        node_colors = 'lightblue'

    fig, ax = plt.subplots()
    nx.draw(G, pos, node_color=node_colors, edge_color='gray', with_labels=True, ax=ax)
    ax.set_title("Gráfstruktúra (fázis színezéssel)")
    st.pyplot(fig)

def draw_3d_graph(G, theta):
    pos = nx.spring_layout(G, dim=3, seed=42)
    xyz = np.array([pos[n] for n in G.nodes()])
    norm_theta = (theta % (2*np.pi)) / (2*np.pi)

    edge_x, edge_y, edge_z = [], [], []
    for i, j in G.edges():
        x0, y0, z0 = pos[i]
        x1, y1, z1 = pos[j]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='none'
    ))

    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=norm_theta,
            colorscale='hsv',
            colorbar=dict(title="Fázis"),
            opacity=0.9
        ),
        text=[f"Node {i}" for i in G.nodes()],
        hoverinfo='text'
    ))

    fig.update_layout(
        title="🌐 3D gráf (fázisszínezéssel)",
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def run():
    st.title("🔗 Gráfalapú szinkronanalízis")
    st.markdown("Kuramoto-modell vizsgálata különböző gráfstruktúrákon, vizuálisan és interaktívan.")

    # 👉 A sidebar mindig aktív legyen, a csúszkákkal együtt
    st.sidebar.header("⚙️ Paraméterek")
    graph_type = st.sidebar.selectbox("Gráftípus", ["Erdős–Rényi", "Kör", "Rács", "Teljes gráf"])
    N = st.sidebar.slider("Csomópontok száma", 5, 100, 30)
    K = st.sidebar.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    steps = st.sidebar.slider("Lépések száma", 10, 1000, 300)
    dt = st.sidebar.slider("Időlépés (dt)", 0.001, 0.1, 0.01)
    er_p = st.sidebar.slider("Erdős–Rényi élvalószínűség", 0.05, 1.0, 0.1, step=0.05)
    show_3d = st.sidebar.checkbox("🌐 3D gráf megjelenítése")

    if st.sidebar.button("▶️ Szimuláció indítása"):
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

        with st.spinner("Szimuláció fut..."):
            start = time.time()
            r_values, theta_hist = run_simulation(G, steps, dt, K)
            end = time.time()

        st.success(f"Szimuláció kész ({end - start:.2f} sec)")
        st.metric("📈 Végső szinkronizációs érték (r)", f"{r_values[-1]:.3f}")
        st.metric("📊 Átlagos r", f"{np.mean(r_values):.3f}")

        st.subheader("📉 Szinkronizáció időbeli alakulása")
        fig1, ax1 = plt.subplots()
        ax1.plot(r_values)
        ax1.set_xlabel("Időlépések")
        ax1.set_ylabel("Szinkronizáció (r)")
        ax1.set_title("Kuramoto-szinkronizáció")
        st.pyplot(fig1)

        st.subheader("🧠 Kezdeti gráf vizualizáció")
        draw_graph(G, theta_hist[0])

        st.subheader("🧠 Végállapot gráf vizualizáció")
        draw_graph(G, theta_hist[-1])

        if show_3d:
            st.subheader("🌐 Interaktív 3D gráf")
            draw_3d_graph(G, theta_hist[-1])

        st.subheader("📝 Jegyzetek")
        notes = st.text_area("Írd le megfigyeléseid vagy ötleteid:", height=150)
        if notes:
            st.download_button("💾 Jegyzet mentése", data=notes, file_name="sync_notes.txt")

    with st.expander("📚 Kuramoto-modell magyarázat"):
        st.markdown("""
        A **Kuramoto-modell** egy klasszikus szinkronizációs modell, ahol oszcillátorok egy gráfhálózaton keresztül hatnak egymásra.
        A szinkronizációs mértéket az **r** paraméter jellemzi, amely 0 (teljesen káoszos) és 1 (teljesen szinkronizált) között mozog.

        - A fázisváltozás képlete:  
          $\\frac{d\\theta_i}{dt} = \\omega_i + \\frac{K}{N} \\sum_j A_{ij} \\sin(\\theta_j - \\theta_i)$

        - **Gráftípusok**:
            - Erdős–Rényi: véletlenszerű élek
            - Kör: szomszédos csomópontok
            - Rács: 2D háló
            - Teljes gráf: minden mindenhez

        Ezen szimuláció segít megérteni, hogyan befolyásolja a gráf szerkezete a szinkronizáció kialakulását.
        """)

# Kötelező ReflectAI-kompatibilitáshoz
app = run
