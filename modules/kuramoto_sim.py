import numpy as np

def run_kuramoto_simulation(n=5, coupling=0.5, timesteps=100):
    theta = np.random.rand(n) * 2 * np.pi
    omega = np.random.rand(n)
    history = [theta.copy()]

    for _ in range(timesteps):
        for i in range(n):
            interaction = np.sum(np.sin(theta - theta[i]))
            theta[i] += omega[i] + (coupling / n) * interaction
        history.append(theta.copy())

    return np.array(history)
