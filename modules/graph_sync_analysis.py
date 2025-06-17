
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def simulate_kuramoto_on_graph(G, timesteps=300, coupling=1.0, noise=0.0):
    n = len(G)
    A = nx.to_numpy_array(G)
    theta = np.random.rand(n) * 2 * np.pi
    omega = np.random.normal(1.0, 0.1, n)
    history = []

    for _ in range(timesteps):
        noise_term = noise * np.random.randn(n)
        theta_dot = omega + noise_term + (coupling / n) * A @ np.sin(np.subtract.outer(theta, theta).T).sum(axis=1)
        theta += theta_dot * 0.05
        history.append(theta.copy())

    history = np.array(history)
    R = np.abs(np.mean(np.exp(1j * history), axis=1))  # Koherencia index

    # Koherencia elérésének időpontja
    threshold = 0.95
    sync_time = np.argmax(R > threshold) if np.any(R > threshold) else len(R)
    return sync_time, R

def compare_graph_topologies(n=20, timesteps=300, noise=0.05):
    graphs = {
        "Random": nx.erdos_renyi_graph(n, 0.2),
        "Small-world": nx.watts_strogatz_graph(n, 4, 0.3),
        "Scale-free": nx.barabasi_albert_graph(n, 3)
    }

    results = {}
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, G in graphs.items():
        sync_time, R = simulate_kuramoto_on_graph(G, timesteps=timesteps, noise=noise)
        results[label] = sync_time
        ax.plot(R, label=f"{label} (T={sync_time})")

    ax.set_title("Szinkronizáció különböző gráftípusokon (Koherencia R(t))")
    ax.set_xlabel("Időlépés")
    ax.set_ylabel("Koherencia")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    return fig, results
