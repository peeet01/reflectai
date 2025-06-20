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
    st.title("🧠 Lorenz Attractor – ESN Predikció (Pro)")

    # 👉 Paraméterek szekció
    with st.sidebar:
        st.header("⚙️ Beállítások")
        steps = st.slider("🔮 Előrejelzendő lépések száma", 1, 50, 20)
        delay = st.slider("⏱️ Beágyazási késleltetés", 1, 10, 3)
        dim = st.slider("📐 Beágyazási dimenzió", 2, 10, 5)

    # Lorenz attraktor generálása
    t_max = 50
    dt = 0.01
    t = np.arange(0, t_max, dt)
    initial_state = [1.0, 1.0, 1.0]
    sol = solve_ivp(lorenz, [0, t_max], initial_state, t_eval=t, method="RK45")
    x, y, z = sol.y

    def embed(data, delay, dim):
        N = len(data) - delay * (dim - 1)
        if N <= 0:
            return np.empty((0, dim))
        return np.array([data[i:i + delay * dim:delay] for i in range(N)])

    X_embed = embed(x, delay, dim)
    y_target = x[delay * dim:delay * dim + len(X_embed)]

    # Védekezés invalid értékek ellen
    if len(X_embed) == 0 or len(y_target) == 0:
        st.error("❌ A beállítások alapján nincs elég adat a modellhez. Próbálj kisebb késleltetést vagy dimenziót.")
        return

    valid_mask = np.all(np.isfinite(X_embed), axis=1) & np.isfinite(y_target)
    X_embed = X_embed[valid_mask]
    y_target = y_target[valid_mask]

    if len(X_embed) < steps + 1:
        st.warning("⚠️ Túl kevés érvényes adat a predikcióhoz. Növeld az időintervallumot vagy csökkentsd az előrejelzett lépések számát.")
        return

    model = Ridge(alpha=1.0)
    model.fit(X_embed[:-steps], y_target[:-steps])
    y_pred = model.predict(X_embed[-steps:])
    pred_x = np.concatenate([x[:-(steps)], y_pred])
    rmse = np.sqrt(mean_squared_error(x[-steps:], y_pred))

    st.markdown("---")
    st.markdown(f"### 📉 RMSE: `{rmse:.4f}`")

    # 3D ábra
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Valós pálya', line=dict(color='blue')))
    fig.add_trace(go.Scatter3d(
        x=pred_x[-steps:], y=y[-steps:], z=z[-steps:], 
        mode='lines', name='Predikció', line=dict(color='red')
    ))
    fig.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        height=600,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    st.plotly_chart(fig, use_container_width=True)
