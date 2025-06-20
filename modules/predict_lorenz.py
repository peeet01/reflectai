import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

def generate_lorenz(T=10000, dt=0.01, sigma=10, rho=28, beta=8/3):
    xs, ys, zs = [1.], [1.], [1.]
    for _ in range(T-1):
        x_dot = sigma * (ys[-1] - xs[-1])
        y_dot = xs[-1] * (rho - zs[-1]) - ys[-1]
        z_dot = xs[-1] * ys[-1] - beta * zs[-1]
        xs.append(xs[-1] + x_dot * dt)
        ys.append(ys[-1] + y_dot * dt)
        zs.append(zs[-1] + z_dot * dt)
    return np.array([xs, ys, zs]).T

def prepare_data(data, lag=10):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i:i+lag])
        y.append(data[i+lag])
    return np.array(X), np.array(y)

def run():
    st.subheader("📈 Lorenz rendszer predikció – gépi tanulással")

    T = st.slider("Időlépések száma", 1000, 10000, 3000, step=500)
    lag = st.slider("Késleltetés (lag)", 5, 50, 10)
    alpha = st.slider("Ridge regresszió α", 0.001, 10.0, 1.0)
    note = st.text_input("📝 Megjegyzés (opcionális):")

    data = generate_lorenz(T)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = prepare_data(scaled_data, lag)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Ridge(alpha=alpha)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

    # Hibagörbe
    error = np.mean((y_pred - y_test)**2, axis=1)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(y_test[:, 0], label="Valós X")
    axs[0].plot(y_pred[:, 0], label="Predikált X", linestyle='dashed')
    axs[0].set_title("📊 Lorenz predikció összehasonlítása")
    axs[0].legend()

    axs[1].plot(error, color="red")
    axs[1].set_title("⚠️ Átlagos négyzetes hiba")
    axs[1].set_xlabel("Idő")
    axs[1].set_ylabel("MSE")

    st.pyplot(fig)

    if note:
        st.markdown(f"📌 Megjegyzés: {note}")
