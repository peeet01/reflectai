import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def simulate_kuramoto(G, T, dt, K):
    N = len(G)
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)
    order_parameters = []

    pos = nx.spring_layout(G, seed=42)
    edges = list(G.edges())

    snapshots = []

    for _ in range(T):
        theta_matrix = np.subtract.outer(theta, theta)
        A = nx.to_numpy_array(G)
        coupling = np.sum(A * np.sin(theta_matrix), axis=1)
        theta += (omega + K * coupling) * dt

        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        order_parameters.append(r)

        snapshots.append(theta.copy())

    return snapshots, order_parameters, pos, edges

def plot_order_parameter(order_parameters, dt):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(order_parameters)) * dt, order_parameters, color='green')
    ax.set_title("Szinkronizációs index alakulása")
    ax.set_xlabel("Idő")
    ax.set_ylabel("r (order parameter)")
    return fig

def plot_network_plotly(pos, theta, edges):
    node_x, node_y, node_z = [], [], []
    node_color = []

    for node in pos:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(0)  # síkban ábrázoljuk
        node_color.append(theta[node])

    edge_x, edge_y, edge_z = [], [], []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [0, 0, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='none'
    )

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=8,
            color=node_color,
            colorscale='hsv',
            colorbar=dict(title="Fázis"),
            line=dict(width=0.5, color='black')
        ),
        hoverinfo='text',
        text=[f'Csomópont {i}' for i in range(len(pos))]
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Szinkronizált hálózat – fázisok színekkel",
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
    )
    return fig

def run():
    st.subheader("🔗 Szinkronizáció hálózatokon (Pro)")

    network_type = st.selectbox("Hálózati topológia", ["rács", "kis világ", "skálafüggetlen", "Erdős–Rényi"])
    N = st.slider("Csomópontok száma", 10, 100, 30)
    T = st.slider("Iterációk száma", 50, 500, 200)
    dt = st.slider("Időlépés (dt)", 0.01, 0.1, 0.05)
    K = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)

    if network_type == "rács":
        G = nx.grid_2d_graph(int(np.sqrt(N)), int(np.sqrt(N)))
        G = nx.convert_node_labels_to_integers(G)
    elif network_type == "kis világ":
        G = nx.watts_strogatz_graph(N, k=4, p=0.2)
    elif network_type == "skálafüggetlen":
        G = nx.barabasi_albert_graph(N, m=2)
    else:
        G = nx.erdos_renyi_graph(N, p=0.1)

    with st.spinner("Szimuláció..."):
        snapshots, order_parameters, pos, edges = simulate_kuramoto(G, T, dt, K)

    st.markdown("### 📉 Szinkronizáció mértéke időben")
    fig_sync = plot_order_parameter(order_parameters, dt)
    st.pyplot(fig_sync)

    st.markdown("### 🌐 Hálózat állapota a szimuláció végén")
    final_fig = plot_network_plotly(pos, snapshots[-1], edges)
    st.plotly_chart(final_fig, use_container_width=True)
