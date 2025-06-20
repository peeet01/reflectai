import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go

def simulate_kuramoto_on_graph(G, K=2.0, T=200, dt=0.05):
    N = len(G.nodes)
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)

    adjacency = nx.to_numpy_array(G)

    for _ in range(T):
        theta_matrix = np.subtract.outer(theta, theta)
        coupling = np.sum(adjacency * np.sin(theta_matrix), axis=1)
        theta += (omega + (K / N) * coupling) * dt

    return theta % (2 * np.pi)

def plot_network_plotly(G, theta, title):
    pos = nx.spring_layout(G, dim=3, seed=42)
    nodes_xyz = np.array([pos[i] for i in G.nodes])
    edges = list(G.edges)

    edge_x, edge_y, edge_z = [], [], []
    for u, v in edges:
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter3d(
        x=nodes_xyz[:, 0],
        y=nodes_xyz[:, 1],
        z=nodes_xyz[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=theta,
            colorscale='hsv',
            colorbar=dict(title='Fázis'),
            line=dict(width=0.5, color='black')
        ),
        hoverinfo='text',
        text=[f"Node {i}<br>Fázis: {theta[i]:.2f}" for i in G.nodes]
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=80, b=20),  # fontos: felső margó
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def run():
    st.subheader("🧠 Topológiai gráf szinkronizáció – színes fázisokkal")

    N = st.slider("🔢 Csúcsok száma", 10, 100, 30)
    p = st.slider("📡 Élképzési valószínűség", 0.05, 1.0, 0.2)
    K = st.slider("🔗 Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    T = st.slider("⏱️ Iterációk száma", 10, 500, 200)
    dt = st.slider("🕒 Időlépés (dt)", 0.01, 0.1, 0.05)

    G = nx.erdos_renyi_graph(N, p)
    final_theta = simulate_kuramoto_on_graph(G, K, T, dt)

    st.markdown("### 🌐 Szinkronizált gráf színezett fázisokkal")
    plot_network_plotly(G, final_theta, title="🌈 Fázisszinkronizáció a gráfban")
