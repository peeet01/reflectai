
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def generate_initial_graph(num_nodes):
    adjacency = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < 0.1:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
    return adjacency


def dynamic_update_graph(adjacency, phases, threshold=0.9):
    num_nodes = len(phases)
    new_adj = np.copy(adjacency)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            sync = np.cos(phases[i] - phases[j])
            if sync > threshold:
                new_adj[i, j] = 1
                new_adj[j, i] = 1
            else:
                new_adj[i, j] *= 0.99
                new_adj[j, i] *= 0.99
    return new_adj


def simulate_generative_kuramoto(N, T, dt, K):
    theta = np.random.uniform(0, 2*np.pi, N)
    omega = np.random.normal(0, 1, N)
    adjacency = generate_initial_graph(N)

    history = []

    for t in range(T):
        dtheta = np.zeros(N)
        for i in range(N):
            interaction = 0
            for j in range(N):
                if adjacency[i, j] > 0:
                    interaction += np.sin(theta[j] - theta[i])
            dtheta[i] = omega[i] + (K / N) * interaction
        theta += dtheta * dt
        adjacency = dynamic_update_graph(adjacency, theta)
        history.append(np.copy(theta))

    return np.array(history)


def run():
    st.header("🌱 Generatív Kuramoto Modell")
    N = st.slider("Oszcillátorok száma", 5, 50, 10)
    T = st.slider("Időlépések száma", 50, 500, 200)
    dt = st.slider("Időlépés (dt)", 0.001, 0.1, 0.01)
    K = st.slider("Kapcsolódási erősség (K)", 0.0, 10.0, 2.0)

    if st.button("Szimuláció futtatása"):
        result = simulate_generative_kuramoto(N, T, dt, K)

        fig, ax = plt.subplots()
        for i in range(N):
            ax.plot(np.unwrap(result[:, i]), label=f"Oszc. {i+1}")
        ax.set_title("Generatív Kuramoto szinkronizáció")
        ax.set_xlabel("Időlépés")
        ax.set_ylabel("Fázis")
        st.pyplot(fig)
