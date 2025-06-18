
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def run_xor_prediction_with_mlp():
    X = np.array([[0,0],[0,1],[1,0],[1,1]] * 100)
    y = np.array([0,1,1,0] * 100)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=300, learning_rate_init=0.01, random_state=42)
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)

    losses = clf.loss_curve_
    fig, ax = plt.subplots()
    ax.plot(losses, label="Loss curve")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("XOR MLP Training Loss")
    ax.legend()
    return fig, accuracy
