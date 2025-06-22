import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

from modules.data_upload import get_uploaded_data, show_data_overview


def lorenz(x, y, z, s=10, r=28, b=2.667):
    dx = s * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z
    return dx, dy, dz


def generate_lorenz_data(dt, steps):
    xs, ys, zs = np.empty(steps), np.empty(steps), np.empty(steps)
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    for i in range(1, steps):
        dx, dy, dz = lorenz(xs[i-1], ys[i-1], zs[i-1])
        xs[i] = xs[i-1] + dx * dt
        ys[i] = ys[i-1] + dy * dt
        zs[i] = zs[i-1] + dz * dt
    return xs, ys, zs


def embed_data(series, delay, dimension, pred_steps=1):
    N = len(series)
    X, y = [], []
    for i in range(N - delay * (dimension + pred_steps - 1)):
        x_i = [series[i + j * delay] for j in range(dimension)]
        y_i = series[i + delay * dimension + (pred_steps - 1)]
        if np.all(np.isfinite(x_i)) and np.isfinite(y_i):
            X.append(x_i)
            y.append(y_i)
    return np.array(X), np.array(y)


def run():
    st.title("🔮 Lorenz attractor predikció Ridge modellel")

    dt = st.slider("Időlépés (dt)", 0.001, 0.05, 0.03)
    steps = st.slider("Időlépések száma", 100, 2000, 1500)
    delay = st.slider("Késleltetés (delay)", 1, 20, 3)
    dimension = st.slider("Beágyazás dimenziója", 2, 10, 5)
    pred_steps = st.slider("Előrejelzendő lépések", 1, 50, 1)

    df = get_uploaded_data(required_columns=["x"], allow_default=True, default="lorenz")

    if df is not None:
        show_data_overview(df)
        x = df["x"].values[:steps]
        y = df["y"].values[:steps] if "y" in df.columns else np.zeros_like(x)
        z = df["z"].values[:steps] if "z" in df.columns else np.zeros_like(x)
    else:
        st.error("❌ Nem áll rendelkezésre megfelelő adat.")
        return

    target_x = x[delay * dimension:]
    X_data, y_data = embed_data(x, delay, dimension, pred_steps)

    if not np.isfinite(X_data).all() or not np.isfinite(y_data).all():
        st.error("❌ Az adatok NaN vagy végtelen értékeket tartalmaznak.")
        return

    if X_data.shape[0] == 0:
        st.warning("⚠️ Túl kevés adat. Csökkentsd a késleltetést vagy dimenziót.")
        return

    model = Ridge()
    model.fit(X_data, y_data)
    pred_x = model.predict(X_data)

    min_len = min(len(target_x), len(pred_x))
    rmse = np.sqrt(mean_squared_error(target_x[:min_len], pred_x[:min_len]))
    st.markdown(f"📉 **RMSE**: `{rmse:.4f}`")

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(x, y, z, lw=0.5)
    ax.set_title("🌪️ Lorenz attraktor")

    ax2 = fig.add_subplot(122)
    ax2.plot(range(len(target_x)), target_x, label="Valós")
    ax2.plot(range(len(pred_x)), pred_x, label="Predikció", alpha=0.7)
    ax2.set_title("🔮 Predikció vs Valós")
    ax2.legend()

    st.pyplot(fig)
