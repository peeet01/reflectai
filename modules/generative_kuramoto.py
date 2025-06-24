import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_graph(n_nodes, p):
    return nx.erdos_renyi_graph(n_nodes, p)

def simulate_kuramoto(G, K, t_max=10, dt=0.05):
    N = len(G.nodes)
    A = nx.to_numpy_array(G)
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(1.0, 0.1, N)
    t = np.arange(0, t_max, dt)
    sync = []

    for _ in t:
        dtheta = omega + K / N * np.sum(A * np.sin(theta[:, None] - theta), axis=1)
        theta += dtheta * dt
        order_param = np.abs(np.sum(np.exp(1j * theta)) / N)
        sync.append(order_param)

    return t, sync, G

def plot_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    st.pyplot(plt.gcf())
    plt.clf()

def plot_sync(t, sync):
    plt.plot(t, sync)
    plt.xlabel("Idő")
    plt.ylabel("Szinkronizáció (r)")
    plt.title("Kuramoto szinkronizációs dinamikája")
    st.pyplot(plt.gcf())
    plt.clf()

def run():
    st.header("🎲 Generatív Kuramoto hálózat")
    st.write("Hozz létre véletlenszerű hálózatokat, és vizsgáld a Kuramoto modell szinkronizációs viselkedését.")

    n_nodes = st.slider("Csomópontok száma", 5, 100, 20)
    p = st.slider("Élképzési valószínűség (p)", 0.0, 1.0, 0.1)
    K = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)

    if st.button("Szimuláció indítása"):
        G = generate_graph(n_nodes, p)
        t, sync, G = simulate_kuramoto(G, K)
        
        st.subheader("🧠 Generált gráf")
        plot_graph(G)

        st.subheader("📈 Szinkronizációs dinamika")
        plot_sync(t, sync)
