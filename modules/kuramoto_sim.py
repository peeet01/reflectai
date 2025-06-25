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
    st.markdown("A kapcsolati erősség most színekkel van vizualizálva a szálak mentén.")

    N = st.slider("🌱 Oszcillátorok száma", 5, 80, 30)
    K = st.slider("💫 Kapcsolódási erősség", 0.0, 10.0, 3.0, 0.1)
    steps = st.slider("⏱️ Iterációk száma", 100, 1500, 500, 100)
    dt = 0.05

    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)
    A = np.ones((N, N)) - np.eye(N)

    order_params = []
    for _ in range(steps):
        theta = kuramoto_step(theta, K, A, omega, dt)
        order_params.append(compute_order_parameter(theta))

    # 🌼 Pitypang elrendezés - körsugár irány
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    radius = 1.5
    pos = {
        i: [
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.random.uniform(-0.2, 0.2)
        ] for i, angle in enumerate(angles)
    }

    node_x, node_y, node_z = zip(*[pos[n] for n in range(N)])
    center = [0, 0, 0]

    fig = go.Figure()

    # 🎨 Színezett szálak a fáziskülönbség alapján
    for i in range(N):
        x1, y1, z1 = pos[i]
        phase_diff = np.abs(np.sin(theta[i] - np.mean(theta)))
        color_val = int(255 * phase_diff)
        fig.add_trace(go.Scatter3d(
            x=[center[0], x1, None],
            y=[center[1], y1, None],
            z=[center[2], z1, None],
            mode='lines',
            line=dict(
                color=f'rgba({color_val}, 0, {255 - color_val}, 0.6)',
                width=2
            ),
            showlegend=False
        ))

    # 🌸 Pitypang-virág oszcillátor pontok
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=6,
            color='black',
            symbol='cross',
            opacity=0.9
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
        font=dict(color='black'),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Szinkronizációs index (R)")
    st.line_chart(order_params)

    st.text_area("📝 Megjegyzés", placeholder="Figyeld meg, hol van nagy fáziskülönbség a kapcsolatok színéből...")

# Kötelező ReflectAI-hoz
app = run
