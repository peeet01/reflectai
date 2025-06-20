import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_topology(N, topology):
    if topology == "rács":
        G = nx.grid_2d_graph(int(np.sqrt(N)), int(np.sqrt(N)))
        G = nx.convert_node_labels_to_integers(G)
    elif topology == "kis világ":
        G = nx.watts_strogatz_graph(N, k=4, p=0.1)
    elif topology == "skálafüggetlen":
        G = nx.barabasi_albert_graph(N, m=2)
    else:
        G = nx.erdos_renyi_graph(N, p=0.1)
    return G


def plot_network(G, phases, title):
    pos = nx.spring_layout(G, seed=42)
    phases = np.array(phases)
    if len(phases) != len(G.nodes):
        phases = phases[:len(G.nodes)]
    nx.draw(
        G, pos, node_color=phases, cmap='hsv',
        node_size=200, with_labels=False
    )
    plt.title(title)


def kuramoto_topology_sim(G, K, steps):
    N = len(G.nodes)
    A = nx.to_numpy_array(G)
    theta = np.random.uniform(0, 2 * np.pi, N)
    dt = 0.05
    history = []

    for _ in range(steps):
        theta_matrix = np.subtract.outer(theta, theta)
        if A.shape != theta_matrix.shape:
            A = A[:theta_matrix.shape[0], :theta_matrix.shape[1]]
        coupling = np.sum(A * np.sin(theta_matrix), axis=1)
        theta += (K / N) * coupling * dt
        history.append(theta.copy())

    return theta, history


def run():
    st.subheader("🧭 Topológiai szinkronizáció – hálózat alapú Kuramoto szimuláció")

    N = st.slider("🔢 Csúcsok száma", 10, 100, 30)
    K = st.slider("📡 Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    steps = st.slider("⏱️ Iterációk száma", 10, 500, 200)
    topology = st.selectbox("🌐 Hálózati topológia", ["rács", "kis világ", "skálafüggetlen", "véletlenszerű"])

    G = generate_topology(N, topology)
    theta_final, _ = kuramoto_topology_sim(G, K, steps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plt.sca(axes[0])
    plot_network(G, np.random.uniform(0, 2 * np.pi, N), "Kezdeti állapot")

    plt.sca(axes[1])
    plot_network(G, theta_final, "Végső állapot")

    st.pyplot(fig)
