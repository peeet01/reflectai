# modules/topo_protect.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def run():
    st.header("🔗 Topológiai szinkronizáció vizsgálata")

    st.markdown("Ez a szimuláció bemutatja, hogyan hat a hálózat topológiája a Kuramoto-szinkronizációra.")

    N = st.slider("Oszcillátorok száma", 5, 100, 30)
    K = st.slider("Kapcsolási erősség (K)", 0.0, 5.0, 1.5, step=0.1)
    topology = st.selectbox("Hálózat topológiája", ["Kör", "Teljes", "Véletlen", "Kis világ", "Rács"])

    # Hálózat generálás
    if topology == "Kör":
        G = nx.cycle_graph(N)
    elif topology == "Teljes":
        G = nx.complete_graph(N)
    elif topology == "Véletlen":
        p = st.slider("Véletlen gráf élsűrűség", 0.1, 1.0, 0.3)
        G = nx.erdos_renyi_graph(N, p)
    elif topology == "Kis világ":
        k = st.slider("Kis világ: szomszédok száma", 2, N-1, 4)
        p = st.slider("Újrahuzalozási valószínűség", 0.0, 1.0, 0.1)
        G = nx.watts_strogatz_graph(N, k, p)
    elif topology == "Rács":
        d = int(np.sqrt(N))
        G = nx.grid_2d_graph(d, d)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

    # Szimulációs paraméterek
    steps = 100
    dt = 0.05
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)

    A = nx.to_numpy_array(G)
    r_vals = []

    for t in range(steps):
        dtheta = omega + (K / N) * np.sum(A * np.sin(np.subtract.outer(theta, theta)), axis=1)
        theta += dt * dtheta
        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        r_vals.append(r)

    # Eredmények megjelenítése
    fig, ax = plt.subplots()
    ax.plot(r_vals)
    ax.set_title("Szinkronizációs mutató (r) időben")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("r")
    st.pyplot(fig)

    # Gráf vizualizálás
    fig2, ax2 = plt.subplots()
    nx.draw_circular(G, node_color=theta, cmap=plt.cm.hsv, with_labels=False, node_size=100, ax=ax2)
    ax2.set_title("Hálózat topológiája – Fázis színezéssel")
    st.pyplot(fig2)
