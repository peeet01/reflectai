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
    st.header("🌼 Kuramoto Pitypang Szinkronizáció")
    st.markdown("Egy vizuálisan természetes, pitypang-stílusú megjelenítés szinkronizáló oszcillátorokkal.")

    N = st.slider("🌱 Oszcillátorok száma", 5, 80, 30)
    K = st.slider("💫 Kapcsolódás erőssége", 0.0, 10.0, 3.0, 0.1)
    steps = st.slider("⏱️ Iterációk", 100, 1500, 500, 100)
    dt = 0.05

    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)
    A = np.ones((N, N)) - np.eye(N)

    order_params = []
    for _ in range(steps):
        theta = kuramoto_step(theta, K, A, omega, dt)
        order_params.append(compute_order_parameter(theta))

    # Sugaras elrendezés: pitypang-hatás
    G = nx.complete_graph(N)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.5
    pos = {
        i: [
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.random.uniform(-0.2, 0.2)
        ] for i, angle in enumerate(angles)
    }

    node_x, node_y, node_z = zip(*[pos[n] for n in G.nodes()])
    edge_x, edge_y, edge_z = [], [], []

    center = [0, 0, 0]
    for i in range(N):
        x1, y1, z1 = pos[i]
        edge_x += [center[0], x1, None]
        edge_y += [center[1], y1, None]
        edge_z += [center[2], z1, None]

    fig = go.Figure()

    # 🌾 Szálak a középpontból – sugarak
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=1),
        opacity=0.5,
        name='Szálak'
    ))

    # 🌼 „Pitypang szirmok” – kis kereszt vagy csillag forma
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=4,
            color='black',
            symbol='cross',
            opacity=0.8
        ),
        name='Sziromcsomópontok'
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
        font=dict(color='black'),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌡️ Szinkronizációs index (R)")
    st.line_chart(order_params)

    st.text_area("📝 Megjegyzés", placeholder="Figyeld meg a természetes formák szinkronizációját...")

# Kötelező ReflectAI-hoz
app = run
