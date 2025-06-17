import numpy as np
import matplotlib.pyplot as plt

def run_kuramoto_simulation(n=10, steps=30, coupling=0.5):
    np.random.seed(42)
    phases = np.random.uniform(0, 2*np.pi, n)
    natural_freqs = np.random.normal(1.0, 0.1, n)

    history = []

    for _ in range(steps):
        phase_diffs = phases[:, None] - phases
        coupling_term = coupling * np.sum(np.sin(phase_diffs), axis=1) / n
        phases += 0.1 * (natural_freqs + coupling_term)
        history.append(phases.copy())

    history = np.array(history)

    fig, ax = plt.subplots()
    for i in range(n):
        ax.plot(history[:, i], label=f'Oszc. {i+1}')
    ax.set_title("Kuramoto-fázisszinkronizáció")
    ax.set_xlabel("Iteráció")
    ax.set_ylabel("Fázis")
    return fig