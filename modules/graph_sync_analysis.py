import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def kuramoto_step(theta, omega, A, K, dt):
    N = len(theta)
    dtheta = omega + (K / N) * np.sum(A * np.sin(np.subtract.outer(theta, theta)), axis=1)
    return theta + dtheta * dt

def run_simulation(G, steps, dt, K):
    N = len(G)
    theta = np.random.rand(N) * 2 * np.pi
    omega = np.random.randn(N)
    A = nx.to_numpy_array(G)

    r_values = []
    for _ in range(steps):
        theta = kuramoto_step(theta, omega, A, K, dt)
        r = np.abs(np.mean(np.exp(1j * theta)))
        r_values.append(r)
    return r_values

def run():
    st.header("🔗 Gráfalapú szinkronanalízis")
    st.write("Kuramoto-modell alkalmazása különböző gráfstruktúrákon.")

    graph_type = st.selectbox("Gráftípus", ["Erdős–Rényi", "Kör", "Rács", "Teljes gráf"])
    N = st.slider("Csomópontok száma", 5, 100, 30)
    K = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    steps = st.slider("Lépések száma", 10, 1000, 300)
    dt = st.slider("Időlépés (dt)", 0.001, 0.1, 0.01)

    if st.button("Szimuláció indítása"):
        if graph_type == "Erdős–Rényi":
            G = nx.erdos_renyi_graph(N, 0.1)
        elif graph_type == "Kör":
            G = nx.cycle_graph(N)
        elif graph_type == "Rács":
            side = int(np.sqrt(N))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
        elif graph_type == "Teljes gráf":
            G = nx.complete_graph(N)

        r_values = run_simulation(G, steps, dt, K)
        fig, ax = plt.subplots()
        ax.plot(r_values)
        ax.set_title("Szinkronizáció mértéke időben")
        ax.set_xlabel("Időlépések")
        ax.set_ylabel("r")
        st.pyplot(fig)
