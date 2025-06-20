import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import streamlit as st
import plotly.graph_objects as go

def run():
    st.subheader("🔮 ESN predikció vizualizáció (Pro)")

    # Paraméterek
    T = st.slider("Szimuláció hossza (időlépések)", 200, 2000, 500)
    dt = st.slider("Időlépés (dt)", 0.001, 0.05, 0.01)
    delay = st.slider("Késleltetés (delay)", 1, 20, 4)
    embed_dim = st.slider("Beágyazás dimenzió", 2, 10, 4)
    predict_steps = st.slider("Előrejelzendő lépések száma", 1, 50, 10)

    # Lorenz rendszer
    sigma, rho, beta = 10, 28, 8/3
    def lorenz(x, y, z):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    def simulate_lorenz(T, dt):
        xs, ys, zs = [1.], [1.], [1.]
        for _ in range(T):
            dx, dy, dz = lorenz(xs[-1], ys[-1], zs[-1])
            xs.append(xs[-1] + dx*dt)
            ys.append(ys[-1] + dy*dt)
            zs.append(zs[-1] + dz*dt)
        return np.array(xs), np.array(ys), np.array(zs)

    x, y, z = simulate_lorenz(T + 200, dt)

    # Beágyazás
    def time_delay_embed(data, delay, dim):
        N = len(data) - (dim - 1) * delay
        if N <= 0:
            return np.empty((0, dim))
        return np.array([data[i:i + dim * delay:delay] for i in range(N)])

    X_embed = time_delay_embed(x, delay, embed_dim)
    y_target = x[(embed_dim * delay):(embed_dim * delay + len(X_embed))]

    # Szűrés: csak ahol minden érték véges és a méretek stimmelnek
    min_len = min(len(X_embed), len(y_target))
    X_embed = X_embed[:min_len]
    y_target = y_target[:min_len]

    valid_mask = (
        np.all(np.isfinite(X_embed), axis=1)
        & np.isfinite(y_target)
    )
    X_valid = X_embed[valid_mask]
    y_valid = y_target[valid_mask]

    if len(X_valid) < 10:
        st.warning("⚠️ Túl kevés érvényes adat. Próbálj más beállításokat.")
        return

    model = Ridge(alpha=1.0)
    model.fit(X_valid, y_valid)
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

    # 3D Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x[:len(y_valid)], y=y[:len(y_valid)], z=z[:len(y_valid)],
                               mode='lines', name='Valós pálya',
                               line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter3d(x=y_pred[:-1], y=y_pred[1:], z=y_valid[1:],
                               mode='lines', name='Predikció',
                               line=dict(color='red', width=2)))
    fig.update_layout(title=f"🌪️ Lorenz attractor (ESN predikció) – RMSE: {rmse:.4f}",
                      scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
                      margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ RMSE: {rmse:.4f}")
