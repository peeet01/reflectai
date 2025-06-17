import numpy as np
import matplotlib.pyplot as plt

def run_kuramoto_simulation(n=10, coupling=0.5, timesteps=200, threshold=0.99):
    theta = np.random.rand(n) * 2 * np.pi
    omega = np.random.normal(1.0, 0.1, n)
    history = []

    steps_to_sync = timesteps

    for t in range(timesteps):
        theta += omega + (coupling / n) * np.sum(np.sin(np.subtract.outer(theta, theta)), axis=1)
        theta = np.mod(theta, 2 * np.pi)
        history.append(theta.copy())

        order_param = np.abs(np.mean(np.exp(1j * theta)))
        if order_param > threshold:
            steps_to_sync = t + 1
            break

    history = np.array(history)

    fig, ax = plt.subplots()
    for i in range(n):
        ax.plot(history[:, i], label=f'Oszc. {i+1}')
    ax.set_title("Kuramoto fázisszinkronizáció")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("Fázis (radian)")
    ax.legend(loc='upper right', fontsize="small")

    return fig, steps_to_sync
