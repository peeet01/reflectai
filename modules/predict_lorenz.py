import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def generate_lorenz(dt, steps, sigma=10.0, beta=8/3, rho=28.0):
    xyz = np.empty((steps + 1, 3))
    xyz[0] = (0., 1., 1.05)
    for i in range(steps):
        x, y, z = xyz[i]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        xyz[i + 1] = xyz[i] + dt * np.array([dx, dy, dz])
    return xyz

def run():
    st.subheader("📈 Lorenz attractor predikció")

    dt = st.slider("Időlépés (dt)", 0.001, 0.05, 0.03)
    steps = st.slider("Időlépések száma", 100, 2000, 1500)
    delay = st.slider("Késleltetés (delay)", 1, 20, 3)
    dimension = st.slider("Beágyazás dimenziója", 2, 10, 5)
    predict_ahead = st.slider("Előrejelzendő lépések", 1, 50, 1)

    data = generate_lorenz(dt, steps)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    max_index = len(data) - delay * (dimension - 1) - predict_ahead
    if max_index <= 10:
        st.warning("⚠️ Túl kevés adat keletkezett ehhez a beállításhoz.")
        return

    # Késleltetett embedding
    embedded = []
    targets = []
    for i in range(max_index):
        window = []
        for j in range(dimension):
            window.extend(data[i + j * delay])
        target_idx = i + delay * (dimension - 1) + predict_ahead
        embedded.append(window)
        targets.append(data[target_idx])

    X_data = np.array(embedded)
    y_data = np.array(targets)

    # Véges adatok ellenőrzése
    finite_mask = np.isfinite(X_data).all(axis=1) & np.isfinite(y_data).all(axis=1)
    X_data = X_data[finite_mask]
    y_data = y_data[finite_mask]

    if len(X_data) < 10:
        st.error("❌ Túl kevés érvényes adat áll rendelkezésre. Próbálj kisebb dimenzióval vagy hosszabb időlépéssel.")
        return

    # Modell
    model = Ridge()
    model.fit(X_data, y_data)
    y_pred = model.predict(X_data)

    # RMSE kiírás
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))
    st.markdown(f"📉 **Gyök-négyzetes átlagos hiba (RMSE):** {rmse:.4f}")

    # 3D vizualizáció
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(y_data[:, 0], y_data[:, 1], y_data[:, 2], label="Valódi", alpha=0.6)
    ax1.set_title("🎯 Valódi trajektória")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], label="Előrejelzett", color="orange", alpha=0.6)
    ax2.set_title("🔮 Predikált trajektória")

    st.pyplot(fig)
