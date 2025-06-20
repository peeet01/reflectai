import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def generate_network(topology, N):
    if topology == "rács":
        side = int(np.sqrt(N))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
    elif topology == "kis világ":
        G = nx.watts_strogatz_graph(N, k=4, p=0.3)
    elif topology == "skálafüggetlen":
        G = nx.barabasi_albert_graph(N, m=2)
    else:
        G = nx.erdos_renyi_graph(N, 0.1)
    return G

def kuramoto_dynamics(G, K, iterations, dt=0.05):
    N = G.number_of_nodes()
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)
    adjacency = nx.to_numpy_array(G)

    for _ in range(iterations):
        theta_diff = np.subtract.outer(theta, theta)
        coupling = np.sum(adjacency * np.sin(theta_diff), axis=1)
        theta += (omega + (K / N) * coupling) * dt
    return theta

def plot_network(G, theta, title):
    pos = nx.spring_layout(G, seed=42)
    node_list = list(G.nodes())
    if len(theta) != len(node_list):
        theta = np.random.uniform(0, 2*np.pi, len(node_list))
    colors = np.angle(np.exp(1j * theta))
    nx.draw(G, pos, node_color=colors, cmap='hsv', node_size=200, with_labels=False)
    plt.title(title)

def run():
    st.subheader("🧭 Topológiai szinkronizáció – hálózat alapú Kuramoto szimuláció")

    N = st.slider("🔢 Csúcsok száma", 10, 100, 30)
    K = st.slider("📡 Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    iterations = st.slider("⏱️ Iterációk száma", 10, 500, 200)
    topology = st.selectbox("🌐 Hálózati topológia", ["rács", "kis világ", "skálafüggetlen", "véletlen"])

    G = generate_network(topology, N)
    initial_theta = np.random.uniform(0, 2 * np.pi, G.number_of_nodes())
    final_theta = kuramoto_dynamics(G, K, iterations)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.sca(axes[0])
    plot_network(G, initial_theta, "Kezdeti állapot")
    plt.sca(axes[1])
    plot_network(G, final_theta, "Végső állapot")

    st.pyplot(fig)
