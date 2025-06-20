import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def plot_network(G, phases, title):
    pos = nx.spring_layout(G, seed=42)
    norm = Normalize(vmin=0, vmax=2*np.pi)
    colors = cm.hsv(norm(phases))

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=colors, ax=ax, node_size=200, cmap='hsv')
    ax.set_title(title)
    ax.axis('off')
    return fig

def kuramoto_topology_sim(N, K, iterations, topology="grid"):
    if topology == "grid":
        size = int(np.sqrt(N))
        G = nx.grid_2d_graph(size, size)
        G = nx.convert_node_labels_to_integers(G)
    elif topology == "small_world":
        G = nx.watts_strogatz_graph(N, k=4, p=0.1)
    elif topology == "random":
        G = nx.erdos_renyi_graph(N, p=0.1)
    else:
        G = nx.path_graph(N)

    A = nx.to_numpy_array(G)
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(0, 1, N)

    dt = 0.05
    for _ in range(iterations):
        theta_matrix = np.subtract.outer(theta, theta)
        coupling = np.sum(A * np.sin(theta_matrix), axis=1)
        theta += (omega + (K / N) * coupling) * dt

    order_parameter = np.abs(np.sum(np.exp(1j * theta)) / N)
    return G, theta, order_parameter

def run():
    st.subheader("🧭 Topológiai szinkronizáció – hálózat alapú Kuramoto szimuláció")

    N = st.slider("🔢 Csúcsok száma", 10, 100, 30)
    K = st.slider("📡 Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    iterations = st.slider("⏱️ Iterációk száma", 10, 500, 200)
    topology = st.selectbox("🌐 Hálózati topológia", ["grid", "small_world", "random", "path"])

    G, final_phases, r = kuramoto_topology_sim(N, K, iterations, topology)

    initial_phases = np.random.uniform(0, 2*np.pi, N)
    fig1 = plot_network(G, initial_phases, "🌀 Kezdeti fázisállapot")
    fig2 = plot_network(G, final_phases, f"🔄 Végső állapot\nSzinkronizációs index r = {r:.2f}")

    st.pyplot(fig1)
    st.pyplot(fig2)
    st.markdown(f"**Globális szinkronizációs mutató:** `r = {r:.2f}`")

    # Hálózati jellemzők táblázat
    st.markdown("### 📊 Hálózati jellemzők")
    st.write({
        "Csomópontok száma": G.number_of_nodes(),
        "Élek száma": G.number_of_edges(),
        "Átlagos fokszám": np.mean([d for n, d in G.degree()]),
        "Átlagos klaszterezettség": nx.average_clustering(G),
        "Szinkronizációs index (r)": round(r, 3)
    })
