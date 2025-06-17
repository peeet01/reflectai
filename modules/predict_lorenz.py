import numpy as np
from sklearn.linear_model import Ridge

def predict_lorenz(data, window=5, alpha=1.0):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window].flatten())
        y.append(data[i+window])

    X, y = np.array(X), np.array(y)
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    predictions = model.predict(X)
    # Visszatöltjük kezdeti értékekkel az eredeti hosszra
    pad = np.zeros((window, y.shape[1]))
    return np.vstack((pad, predictions))