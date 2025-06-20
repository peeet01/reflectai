import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import streamlit as st

def lorenz(x, y, z, sigma=10.0, beta=8/3, rho=28.0):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def run():
    st.subheader("📈 Lorenz attractor predikció")

    dt = st.slider("Időlépés (dt)", 0.001, 0.05, 0.03)
    steps = st.slider("Időlépések száma", 100, 2000, 1500)

    x, y, z = 0.1, 0.0, 0.0
    xs, ys, zs = [], [], []

    for _ in range(steps):
        dx, dy, dz = lorenz(x, y, z)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs.append(x)
        ys.append(y)
        zs.append(z)

    # Dataset felkészítése
    lag = 10
    X_data = []
    y_data = []

    for i in range(lag, len(xs)):
        features = [xs[i - j] for j in range(1, lag + 1)]
        X_data.append(features)
        y_data.append([xs[i], ys[i], zs[i]])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # Modell tanítása
    model = Ridge(alpha=1.0)
    model.fit(X_data, y_data)

    # Predikció indítása
    last_input = X_data[0]
    pred_x, pred_y, pred_z = [], [], []

    for _ in range(len(y_data)):
        prediction = model.predict([last_input])[0]
        pred_x.append(prediction[0])
        pred_y.append(prediction[1])
        pred_z.append(prediction[2])
        last_input = np.roll(last_input, -1)
        last_input[-1] = prediction[0]

    # Valós értékek
    target_x = y_data[:, 0]
    target_y = y_data[:, 1]
    target_z = y_data[:, 2]

    # Hossz egyeztetés
    min_len = min(len(pred_x), len(target_x))
    pred_x = np.array(pred_x[:min_len])
    pred_y = np.array(pred_y[:min_len])
    pred_z = np.array(pred_z[:min_len])
    target_x = np.array(target_x[:min_len])
    target_y = np.array(target_y[:min_len])
    target_z = np.array(target_z[:min_len])

    # Védekezés NaN vagy végtelen értékek ellen
    def is_valid(arr):
        return np.all(np.isfinite(arr)) and not np.any(np.isnan(arr))

    if not all(map(is_valid, [pred_x, pred_y, pred_z, target_x, target_y, target_z])):
        st.error("❌ A predikció során nem számszerű érték keletkezett (NaN vagy inf). Próbálj más paramétereket.")
        return

    # Hiba számítása
    error = np.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2 + (pred_z - target_z)**2)
    rmse = np.sqrt(mean_squared_error(
        np.vstack([target_x, target_y, target_z]).T,
        np.vstack([pred_x, pred_y, pred_z]).T
    ))

    st.markdown(f"📉 **RMSE (gyökös átlagos négyzetes hiba): {rmse:.4f}**")

    # Ábrázolás
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    axs[0].plot(xs, ys, zs, lw=0.5)
    axs[0].set_title("🌪️ Lorenz attractor pálya")
    axs[0].set_xlabel("X"); axs[0].set_ylabel("Y")

    axs[1].plot(target_x, label="Valós")
    axs[1].plot(pred_x, '--', label="Predikció")
    axs[1].set_title("🔍 X predikció vs. valós")
    axs[1].legend()

    axs[2].plot(error, color='red')
    axs[2].set_title("📊 Predikciós hiba alakulása")
    axs[2].set_xlabel("Lépés"); axs[2].set_ylabel("Hiba")

    st.pyplot(fig)
