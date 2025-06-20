import streamlit as st
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def run():
    st.title("🌀 Lorenz attractor (ESN predikció)")

    # Paraméterek
    delay = st.slider("⏱️ Késleltetés", 1, 10, 3)
    dim = st.slider("📐 Beágyazás dimenziója", 2, 10, 5)
    steps = st.slider("🔮 Előrejelzendő lépések száma", 1, 50, 20)

    # Lorenz generálása
    dt = 0.01
    t = np.arange(0, 50, dt)
    init = [1.0, 1.0, 1.0]
    sol = solve_ivp(lorenz, [0, 50], init, t_eval=t)
    x, y, z = sol.y

    def embed(data, delay, dim):
        N = len(data) - delay * (dim - 1)
        if N <= 0:
            return np.empty((0, dim))
        return np.array([data[i:i + delay * dim:delay] for i in range(N)])

    X_embed = embed(x, delay, dim)
    y_target = x[delay * dim:delay * dim + len(X_embed)]

    if X_embed.shape[0] == 0 or y_target.shape[0] == 0:
        st.error("❗ Nincs elegendő adat a tanításhoz. Csökkentsd a késleltetést vagy a dimenziót.")
        return

    # Ellenőrzés: egyező méret és finom értékek
    min_len = min(len(X_embed), len(y_target))
    X_embed = X_embed[:min_len]
    y_target = y_target[:min_len]

    valid_mask = np.all(np.isfinite(X_embed), axis=1) & np.isfinite(y_target)
    X_valid = X_embed[valid_mask]
    y_valid = y_target[valid_mask]

    if len(X_valid) <= steps:
        st.warning("⚠️ Kevés érvényes adat. Csökkentsd az előrejelzett lépések számát.")
        return

    # Modell és tanítás
    model = Ridge(alpha=1.0)
    model.fit(X_valid[:-steps], y_valid[:-steps])
    y_pred = model.predict(X_valid[-steps:])
    rmse = np.sqrt(mean_squared_error(y_valid[-steps:], y_pred))

    # 3D ábra
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Valós pálya', line=dict(color='blue')))
    fig.add_trace(go.Scatter3d(
        x=x[-steps:], y=y[-steps:], z=z[-steps:],
        mode='lines', name='Predikció', line=dict(color='red')
    ))
    fig.update_layout(
        title=f"🌪️ ESN Predikció – RMSE: {rmse:.4f}",
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        height=600,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)
