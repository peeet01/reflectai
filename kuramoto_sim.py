import numpy as np
import matplotlib.pyplot as plt

def run_kuramoto_simulation(n=10, steps=50, coupling=0.5, threshold=0.95):
    np.random.seed(42)
    phases = np.random.uniform(0, 2*np.pi, n)
    freqs = np.random.normal(1.0, 0.1, n)
    history = []

    for step in range(steps):
        diffs = phases[:, None] - phases
        coupling_term = coupling * np.sum(np.sin(diffs), axis=1) / n
        phases += 0.1 * (freqs + coupling_term)
        history.append(phases.copy())

        if compute_sync_steps(np.array(history)) >= threshold:
            return plot_phases(history), step + 1

    return plot_phases(history), steps

def compute_sync_steps(history):
    order_params = []
    for phase_vector in history:
        complex_order = np.mean(np.exp(1j * phase_vector))
        order_params.append(np.abs(complex_order))
    return order_params[-1]

def plot_phases(history):
    history = np.array(history)
    fig, ax = plt.subplots()
    for i in range(history.shape[1]):
        ax.plot(history[:, i], label=f'Oszc. {i+1}')
    ax.set_title("Kuramoto-fázisszinkronizáció")
    ax.set_xlabel("Iteráció")
    ax.set_ylabel("Fázis")
    return fig