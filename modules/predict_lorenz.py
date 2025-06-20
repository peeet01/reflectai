import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

def lorenz_system(state, sigma=10.0, beta=8/3, rho=28.0):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def generate_lorenz_data(initial_state, dt, steps):
    states = np.zeros((steps, 3))
    states[0] = initial_state
    for i in range(1, steps):
        k1 = lorenz_system(states[i-1])
        k2 = lorenz_system(states[i-1] + dt * k1 / 2)
        k3 = lorenz_system(states[i-1] + dt * k2 / 2)
        k4 = lorenz_system(states[i-1] + dt * k3)
        states[i] = states[i-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return states

def run():
    st.subheader("📈 Lorenz attractor predikció")

    dt = st.slider("Időlépés (dt)", 0.001, 0.05, 0.03)
    steps = st.slider("Időlépések száma", 100, 2000, 1500)
    delay = 1
    dimension = 6

    initial_state = np.array([1.0, 1.0, 1.0])
    data = generate_lorenz_data(initial_state, dt, steps)

    # Ábrázolás: eredeti pálya
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(data[:, 0], data[:, 1], data[:, 2], color='blue', alpha=0.6)
    ax.set_title("🎢 Lorenz attractor (valódi)")

    # Késleltetett beágyazás
    max_index = len(data) - delay * (dimension - 1)
    if max_index <= 0:
        st.error("❌ Túl kevés adat a késleltetett beágyazáshoz. Növeld az időlépések számát vagy csökkentsd a dimenziót.")
        return

    embedded = np.array([
        np.hstack([data[i + j * delay] for j in range(dimension)])
        for i in range(max_index)
    ])

    X_data = embedded[:-1]
    y_data = data[delay * dimension:max_index + delay]

    # Védekezés: eltávolítjuk a NaN/inf adatokat
    valid_mask = np.all(np.isfinite(X_data), axis=1) & np.all(np.isfinite(y_data), axis=1)
    X_data = X_data[valid_mask]
    y_data = y_data[valid_mask]

    if len(X_data) == 0:
        st.error("❌ A tanításhoz nincs elegendő érvényes adat.")
        return

    # Modell tanítása és predikció
    model = Ridge()
    model.fit(X_data, y_data)

    predictions = model.predict(X_data)
    pred_x, pred_y, pred_z = predictions.T
    target_x, target_y, target_z = y_data.T

    rmse = np.sqrt(mean_squared_error(y_data, predictions))

    # Predikció ábrázolása
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(pred_x, pred_y, pred_z, color='red', alpha=0.6)
    ax2.set_title(f"🤖 Predikció (RMSE = {rmse:.4f})")

    st.pyplot(fig)
