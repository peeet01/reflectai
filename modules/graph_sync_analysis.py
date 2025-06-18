import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def run():
    st.subheader("🔬 Topológiai szinkronizációs vizsgálat")
    st.write("Három gráfmodell összehasonlítása szinkronizáció alapján.")

    graphs = {
        "Random": nx.erdos_renyi_graph(30, 0.1),
        "Small-world": nx.watts_strogatz_graph(30, 4, 0.3),
        "Scale-free": nx.barabasi_albert_graph(30, 2)
    }

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, G) in zip(axs, graphs.items()):
        A = nx.to_numpy_array(G)
        eigs = np.linalg.eigvals(A)
        coherence = np.real(eigs).max() / np.real(eigs).sum()
        nx.draw(G, ax=ax, node_size=40, with_labels=False)
        ax.set_title(f"{name}\nKoherencia: {coherence:.2f}")
    st.pyplot(fig)
