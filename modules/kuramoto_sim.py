import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import time

def kuramoto_step(theta, K, A, omega, dt):
    N = len(theta)
    dtheta = omega + (K / N) * np.sum(A * np.sin(theta[:, None] - theta), axis=1)
    return theta + dt * dtheta

def compute_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))

def generate_graph(N, graph_type):
    if graph_type == "Teljes":
        G = nx.complete_graph(N)
    elif graph_type == "Véletlen (Erdős-Rényi)":
        G = nx.erdos_renyi_graph(N, p=0.3)
    elif graph_type == "Kis világ (Watts-Strogatz)":
        G = nx.watts_strogatz_graph(N, k=4, p=0.3)
    elif graph_type == "Skálafüggetlen (Barabási-Albert)":
        G = nx.barabasi_albert_graph(N, m=2)
    else:
        G = nx.complete_graph(N)
    return G

def run():
    st.title("🧠 Kuramoto Szinkronizáció – Interaktív Vizualizáció")

    N = st.slider("Oszcillátorok száma", 5, 100, 30)
    K = st.slider("Kapcsolódási erősség (K)", 0.0, 10.0, 2.0, 0.1)
    steps = st.slider("Iterációk", 100, 2000, 500, 100)
    dt = 0.05

    graph_type = st.selectbox("Hálózat típusa", [
        "Teljes",
        "Véletlen (Erdős-Rényi)",
        "Kis világ (Watts-Strogatz)",
        "Skálafüggetlen (Barabási-Albert)"
    ])
    palette = st.selectbox("Színséma", ["Turbo", "Viridis", "Electric", "Hot", "Rainbow"])

    st.subheader("Szimuláció futtatása")
    progress = st.progress(0)

    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)
    G = generate_graph(N, graph_type)
    A = nx.to_numpy_array(G)

    order_params = []
    for step in range(steps):
        theta = kuramoto_step(theta, K, A, omega, dt)
        order_params.append(compute_order_parameter(theta))
        progress.progress((step + 1) / steps)

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
        opacity=0.5,
        name='Kötések'
    ))

    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=8,
            color=theta,
            colorscale=palette,
            opacity=0.9,
            line=dict(color='black', width=0.5),
            symbol='diamond'
        ),
        name='Oszcillátorok'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='', showgrid=False, zeroline=False),
            yaxis=dict(title='', showgrid=False, zeroline=False),
            zaxis=dict(title='', showgrid=False, zeroline=False),
            bgcolor='white'
        ),
        paper_bgcolor='white',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Szinkronizációs index (R)")
    st.line_chart(order_params)

    st.text_area("Megjegyzés", placeholder="Mit figyeltél meg a szinkronizáció során?")

# ReflectAI integrációhoz kötelező:
app = run
