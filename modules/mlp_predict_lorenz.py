import numpy as np
from sklearn.neural_network import MLPRegressor

def predict_lorenz_mlp(data, window=5, hidden_layer_sizes=(64,), max_iter=500):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window].flatten())
        y.append(data[i+window])
    
    X, y = np.array(X), np.array(y)
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                         max_iter=max_iter, solver='adam', random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)

    pad = np.zeros((window, y.shape[1]))
    return np.vstack((pad, predictions))