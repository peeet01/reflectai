import numpy as np

def run_xor_prediction(epochs=100, learning_rate=0.1):
    np.random.seed(0)
    data = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([0,1,1,0])
    weights = np.random.randn(2)
    bias = 0.0

    correct = 0
    for epoch in range(epochs):
        for i in range(4):
            x = data[i]
            y = labels[i]
            output = 1 if np.dot(x, weights) + bias > 0 else 0
            error = y - output
            weights += learning_rate * error * x
            bias += learning_rate * error
            if output == y:
                correct += 1

    accuracy = (correct / (4 * epochs)) * 100
    return accuracy