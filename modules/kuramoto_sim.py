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
    st.header("💥 Kuramoto Szinkronizáció – Robbanáshatásban")
    st.markdown("Egy sci-fi stílusú, 3D robbanásszerű látványvilág a szinkronizáció vizualizálására.")

    N = st.slider("🧠 Oszcillátorok száma", 5, 80, 30)
    K = st.slider("💣 Kapcsolási erősség", 0.0, 10.0, 3.5, 0.1)
    steps = st.slider("⏱️ Iterációk", 100, 2000, 500, 100)
    dt = 0.05

    palette = st.selectbox("🎨 Színséma", ["Turbo", "Hot", "Electric", "Inferno", "Rainbow"])

    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)
    A = np.ones((N, N)) - np.eye(N)

    order_params = []
    for _ in range(steps):
        theta = kuramoto_step(theta, K, A, omega, dt)
        order_params.append(compute_order_parameter(theta))

    # 💥 Robbanás-hatás: nagyobb szórás a 3D térben
    G = nx.complete_graph(N)
    pos = {i: np.random.normal(scale=2.5, size=3) for i in G.nodes()}
    node_x, node_y, node_z = zip(*[pos[n] for n in G.nodes()])
    edge_x, edge_y, edge_z = [], [], []

    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    fig = go.Figure()

    # 🌌 Halvány, áttetsző élek
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='white', width=1, dash='dot'),
        opacity=0.3,
        name='Energiakapcsolatok'
    ))

    # 💣 Kitörésszerű markerek
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=12,
            color=theta,
            colorscale=palette,
            symbol='circle',
            opacity=0.95,
            line=dict(color='gold', width=2)
        ),
        name='Robbanás-oszcillátorok'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='', showgrid=False, zeroline=False),
            yaxis=dict(title='', showgrid=False, zeroline=False),
            zaxis=dict(title='', showgrid=False, zeroline=False),
            bgcolor='black'
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Szinkronizációs index (R)")
    st.line_chart(order_params)

    st.text_area("📓 Megjegyzés", placeholder="Írd le a megfigyeléseidet a robbanásszerű szinkronizációról...")

# Kötelező ReflectAI-hoz
app = run
