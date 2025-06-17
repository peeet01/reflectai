import numpy as np
import matplotlib.pyplot as plt

def run_kuramoto_simulation(n=5, coupling=0.5, timesteps=100, sync_threshold=0.95):
    theta = np.random.rand(n) * 2 * np.pi
    omega = np.random.rand(n)
    history = [theta.copy()]
    steps_needed = timesteps

    for t in range(timesteps):
        for i in range(n):
            interaction = np.sum(np.sin(theta - theta[i]))
            theta[i] += omega[i] + (coupling / n) * interaction
        history.append(theta.copy())

        order_param = np.abs(np.sum(np.exp(1j * theta)) / n)
        if order_param >= sync_threshold:
            steps_needed = t + 1
            break

    history = np.array(history)
    fig, ax = plt.subplots()
    for i in range(history.shape[1]):
        ax.plot(history[:, i], label=f'Oszcillátor {i+1}')
    ax.set_title("Kuramoto szinkronizáció")
    ax.set_xlabel("Időlépés")
    ax.set_ylabel("Fázis")
    ax.legend()

    return fig, steps_needed