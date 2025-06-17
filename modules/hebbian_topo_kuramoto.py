import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def run_topo_adaptive_kuramoto(N=20, T=200, dt=0.1, K=1.0, eta=0.01):
    np.random.seed(42)
    theta = 2 * np.pi * np.random.rand(N)
    omega = np.random.normal(1.0, 0.1, N)
    W = np.ones((N, N)) - np.eye(N)

    # Kisvilág topológia létrehozása
    G = nx.watts_strogatz_graph(N, k=4, p=0.2)
    A = nx.to_numpy_array(G)
    W *= A  # csak létező élek tanulnak

    theta_history = []
    r_history = []
    W_history = []

    for t in range(T):
        dtheta = np.zeros(N)
        for i in range(N):
            coupling = np.sum(W[i] * np.sin(theta - theta[i]))
            dtheta[i] = omega[i] + (K / N) * coupling

        theta += dtheta * dt
        theta_history.append(theta.copy())

        # Szimmetrikus Hebbian update
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j]:
                    delta = np.cos(theta[i] - theta[j])
                    W[i, j] += eta * delta
                    W[j, i] += eta * delta

        W = np.clip(W, -2, 2)
        W_history.append(W.copy())

        # Koherencia index
        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        r_history.append(r)

    theta_array = np.array(theta_history)
    r_array = np.array(r_history)

    # Vizualizáció
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].imshow(theta_array.T, aspect='auto', cmap='twilight')
    ax[0].set_title("Fázisdinamika (θ időben)")
    ax[0].set_ylabel("Oszcillátor index")

    ax[1].plot(r_array)
    ax[1].set_title("Koherencia index R(t)")
    ax[1].set_xlabel("Időlépés")
    ax[1].set_ylabel("R")

    plt.tight_layout()
    return fig