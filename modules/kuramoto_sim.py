import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go

def kuramoto_step(theta, K, A, omega, dt):
    N = len(theta)
    dtheta = omega + (K / N) * np.sum(A * np.sin(theta[:, None] - theta), axis=1)
    return theta + dt * dtheta

def compute_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))

def run():
    st.header("🕸️ Kuramoto Szinkronizációs Modell")
    st.markdown("Vizualizációs szimuláció oszcillátorok szinkronizációjára.")

    N = st.slider("Oszcillátorok száma", 5, 50, 20)
    K = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0, 0.1)
    steps = st.slider("Iterációk száma", 100, 1000, 300, 50)
    dt = 0.05

    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)
    A = np.ones((N, N)) - np.eye(N)  # Teljes gráf

    order_params = []

    for _ in range(steps):
        theta = kuramoto_step(theta, K, A, omega, dt)
        order_params.append(compute_order_parameter(theta))

    # Hálózat vizualizálása
    G = nx.complete_graph(N)
    pos = nx.spring_layout(G, seed=42, dim=3)

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
        mode='lines', line=dict(color='gray', width=1),
        name='Kötések'
    ))

    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=6, color=theta, colorscale='hsv', colorbar=dict(title='Fázis')),
        name='Oszcillátorok'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis=dict(title=''), yaxis=dict(title=''), zaxis=dict(title=''))
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Szinkronizációs index (R)")
    st.line_chart(order_params)

# Kötelező ReflectAI-hoz
app = run
