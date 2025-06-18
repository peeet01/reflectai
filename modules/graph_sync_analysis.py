import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def run():
    st.subheader("📊 Topológiai szinkronizáció")

    st.write("Random gráf és Small-world gráf szinkronizációs tulajdonságainak összehasonlítása.")

    num_nodes = 20
    k = 4  # kapcsolatok száma
    p = 0.3  # rewiring valószínűsége

    G_random = nx.erdos_renyi_graph(num_nodes, 0.3)
    G_smallworld = nx.watts_strogatz_graph(num_nodes, k, p)

    def kuramoto_sim(G, steps=200, K=0.5):
        A = nx.to_numpy_array(G)
        N = A.shape[0]
        theta = np.random.rand(N) * 2 * np.pi
        dt = 0.05
        history = [theta.copy()]
        for _ in range(steps):
            theta_dot = np.zeros(N)
            for i in range(N):
                interaction = np.sum(A[i, j] * np.sin(theta[j] - theta[i]) for j in range(N))
                theta_dot[i] = K * interaction
            theta += dt * theta_dot
            history.append(theta.copy())
        return np.array(history)

    # Szimulációk
    rand_sync = kuramoto_sim(G_random)
    sw_sync = kuramoto_sim(G_smallworld)

    fig, ax = plt.subplots()
    for i in range(rand_sync.shape[1]):
        ax.plot(rand_sync[:, i], label=f"Rand {i}", alpha=0.5)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    for i in range(sw_sync.shape[1]):
        ax2.plot(sw_sync[:, i], label=f"SW {i}", alpha=0.5)
    st.pyplot(fig2)

    st.success("Szimulációk lefutottak.")
