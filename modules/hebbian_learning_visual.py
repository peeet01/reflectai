import numpy as np
import matplotlib.pyplot as plt

def run_hebbian_learning_with_visual(learning_rate=0.1, noise_level=0.1, iterations=100):
    np.random.seed(42)
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    targets = np.array([[0],[1],[1],[0]])

    weights = np.random.randn(2, 1)
    loss_history = []
    weight_history = []

    for _ in range(iterations):
        noisy_inputs = inputs + noise_level * np.random.randn(*inputs.shape)
        predictions = 1 / (1 + np.exp(-np.dot(noisy_inputs, weights)))
        error = targets - predictions
        weights += learning_rate * np.dot(noisy_inputs.T, error)
        loss = np.mean(np.square(error))
        loss_history.append(loss)
        weight_history.append(weights.copy())

    # Ábra: súlyok változása
    weight_array = np.array(weight_history).squeeze()  # shape: (iter, 2)
    fig, ax = plt.subplots()
    for i in range(weight_array.shape[1]):
        ax.plot(weight_array[:, i], label=f'Súly {i+1}')
    ax.set_xlabel("Iteráció")
    ax.set_ylabel("Súly érték")
    ax.set_title("Hebbian tanulás – súlydinamika")
    ax.legend()
    plt.tight_layout()

    return fig