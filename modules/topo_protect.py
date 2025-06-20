import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st


def kuramoto_dynamics(G, K, T, dt=0.05):
    N = len(G)
    A = nx.to_numpy_array(G)
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(0, 1, N)
    sync_history = []

    for _ in range(T):
        theta_matrix = np.subtract.outer(theta, theta)
        coupling = np.sum(A * np.sin(theta_matrix), axis=1)
        theta += (omega + (K / N) * coupling) * dt

        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        sync_history.append(r)

    return theta, sync_history


def plot_network(G, theta, title):
    pos = nx.spring_layout(G, seed=42)
    colors = np.angle(np.exp(1j * theta))
    nx.draw(G, pos, node_color=colors, cmap='hsv', node_size=200, with_labels=False)
    plt.title(title)


def run():
    st.subheader("🧭 Topológiai szinkronizáció – hálózat alapú Kuramoto szimuláció")

    # Beállítások
    N = st.slider("🔢 Csúcsok száma", 10, 100, 30)
    K = st.slider("📡 Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    T = st.slider("⏱️ Iterációk száma", 10, 500, 200)
    topo = st.selectbox("🌐 Hálózati topológia", ["rács", "kis-világ", "skálafüggetlen"])

    # Hálózat létrehozása
    if topo == "rács":
        side = int(np.sqrt(N))
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
    elif topo == "kis-világ":
        G = nx.watts_strogatz_graph(N, k=4, p=0.3)
    elif topo == "skálafüggetlen":
        G = nx.barabasi_albert_graph(N, m=2)

    # Kuramoto futtatás
    final_theta, sync_history = kuramoto_dynamics(G, K, T)

    # Vizualizáció
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plt.figure()
        plot_network(G, np.random.uniform(0, 2*np.pi, N), "Kezdeti állapot")
        st.pyplot(fig1)

    with col2:
        fig2 = plt.figure()
        plot_network(G, final_theta, "Végső állapot")
        st.pyplot(fig2)

    # Szinkronizációs index
    st.line_chart(sync_history)
    st.success(f"📊 Végső szinkronizációs index: r = {sync_history[-1]:.2f}")
    st.info("A különböző topológiák eltérő mértékű szinkronizációt eredményezhetnek. A kis-világ hálók gyakran hatékonyabbak.")
