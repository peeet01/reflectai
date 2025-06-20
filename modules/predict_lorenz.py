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
        st.warning("⚠️ Túl kevés adat keletkezett ehhez a beállításhoz. Növeld az időlépések számát vagy csökkentsd a dimenziót.")
        return

    # Késleltetett vektorok
    embedded = np.array([
        np.hstack([data[i + j * delay] for j in range(dimension)])
        for i in range(max_index)
    ])
    targets = data[delay * (dimension - 1) + predict_ahead : delay * (dimension - 1) + predict_ahead + max_index]

    # Csak véges adatokkal dolgozzunk
    valid_mask = np.all(np.isfinite(embedded), axis=1) & np.all(np.isfinite(targets), axis=1)
    X_data = embedded[valid_mask]
    y_data = targets[valid_mask]

    if len(X_data) < 10:
        st.error("❌ Nem áll rendelkezésre elég adat a tanításhoz. Próbálj kisebb dimenzióval vagy több időlépéssel.")
        return

    # Modell betanítás
    model = Ridge()
    model.fit(X_data, y_data)
    predictions = model.predict(X_data)

    # Kiértékelés
    rmse = np.sqrt(mean_squared_error(y_data, predictions))
    st.markdown(f"📉 **Gyök-átlag-négyzetes hiba (RMSE):** {rmse:.4f}")

    # 3D ábrák
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(y_data[:, 0], y_data[:, 1], y_data[:, 2], label="Valódi", alpha=0.6)
    ax1.set_title("🎯 Valódi trajektória")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label="Előrejelzett", color="orange", alpha=0.6)
    ax2.set_title("🔮 Predikált trajektória")
    st.pyplot(fig)
