
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def run_kuramoto_hebbian(n=10, coupling=0.5, timesteps=100):
    G = nx.erdos_renyi_graph(n, 0.3)
    A = nx.to_numpy_array(G)
    theta = np.random.rand(n) * 2 * np.pi
    omega = np.random.rand(n)
    history = [theta.copy()]

    for _ in range(timesteps):
        theta_dot = omega + (coupling / n) * A @ np.sin(np.subtract.outer(theta, theta).T).sum(axis=1)
        theta += theta_dot * 0.1
        history.append(theta.copy())
        # Hebbian tanulás (adaptív A)
        A += 0.01 * np.outer(np.sin(theta), np.sin(theta))
        np.fill_diagonal(A, 0)

    history = np.array(history)
    fig, ax = plt.subplots()
    for i in range(n):
        ax.plot(history[:, i])
    ax.set_title("Kuramoto–Hebbian Szinkronizáció")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("Fázis")

    sync_steps = np.argmax(np.std(history, axis=1) < 0.1)
    coherence = 1 - np.std(history[-1])
    return fig, {"sync_steps": sync_steps, "coherence": coherence}
