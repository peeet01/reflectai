import numpy as np

def predict_lorenz(data, window=1):
    # Egyszerű lineáris előrejelzés: x(t+1) = x(t)
    return np.roll(data, -1, axis=0)