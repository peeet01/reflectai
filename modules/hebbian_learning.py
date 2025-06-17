import numpy as np
import matplotlib.pyplot as plt

def run_hebbian_learning_with_noise(learning_rate=0.1, noise_level=0.1, iterations=100):
    np.random.seed(42)
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    targets = np.array([[0],[1],[1],[0]])

    weights = np.random.randn(2, 1)
    loss_history = []

    for _ in range(iterations):
        noisy_inputs = inputs + noise_level * np.random.randn(*inputs.shape)
        predictions = 1 / (1 + np.exp(-np.dot(noisy_inputs, weights)))
        error = targets - predictions
        weights += learning_rate * np.dot(noisy_inputs.T, error)
        loss = np.mean(np.square(error))
        loss_history.append(loss)

    fig, ax = plt.subplots()
    ax.plot(loss_history, label="Hibatörténet")
    ax.set_xlabel("Iteráció")
    ax.set_ylabel("Hiba")
    ax.set_title("Hebbian tanulás zajjal")
    ax.legend()
    plt.tight_layout()
    return fig
