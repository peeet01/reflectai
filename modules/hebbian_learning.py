import numpy as np
import matplotlib.pyplot as plt

def run_hebbian_learning_with_noise(epochs=20, n_inputs=5, learning_rate=0.1, noise_std=0.05):
    np.random.seed(42)
    inputs = np.random.choice([-1, 1], size=(epochs, n_inputs))
    weights = np.zeros(n_inputs)
    weights_history = []

    for epoch in range(epochs):
        x = inputs[epoch]
        y = np.dot(weights, x) + np.random.normal(0, noise_std)
        weights += learning_rate * x * y
        weights_history.append(weights.copy())

    weights_history = np.array(weights_history)
    fig, ax = plt.subplots()
    for i in range(n_inputs):
        ax.plot(weights_history[:, i], label=f'Súly {i+1}')
    ax.set_title("Hebbian tanulás zajjal")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Súly érték")
    ax.legend()
    return fig