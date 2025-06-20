import streamlit as st
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from scipy.integrate import solve_ivp


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


def embed_delay_coordinates(data, delay, dimension):
    N = len(data)
    if N - (dimension - 1) * delay <= 0:
        return np.empty((0, dimension))
    return np.array([data[i:i + delay * dimension:delay] for i in range(N - delay * (dimension - 1))])


def run():
    st.subheader("🌪️ Lorenz attractor (ESN predikció – Pro)")

    delay = st.slider("⏳ Késleltetés", 1, 10, 5)
    embed_dim = st.slider("📐 Beágyazási dimenzió", 2, 10, 5)
    pred_steps = st.slider("🔮 Előrejelzendő lépések száma", 1, 50, 20)

    # Lorenz adatok
    t_max = 50
    dt = 0.01
    t = np.arange(0, t_max, dt)
    sol = solve_ivp(lorenz, [0, t_max], [1.0, 1.0, 1.0], t_eval=t)
    x, y, z = sol.y

    # Beágyazás
    x_embed = embed_delay_coordinates(x, delay, embed_dim)
    y_target = x[delay * embed_dim:]

    valid_mask = np.all(np.isfinite(x_embed), axis=1) & np.isfinite(y_target)
    x_embed = x_embed[valid_mask]
    y_target = y_target[valid_mask]

    if len(x_embed) == 0:
        st.error("❌ Az aktuális késleltetés és dimenzió mellett nincs elegendő adat.")
        return

    # Tanító / teszt szétválasztás
    train_size = int(len(x_embed) * 0.8)
    X_train, y_train = x_embed[:train_size], y_target[:train_size]
    X_test, y_test = x_embed[train_size:], y_target[train_size:]

    # ESN helyett egyszerű Ridge
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    pred = model.predict(X_test[:pred_steps])
    rmse = np.sqrt(mean_squared_error(y_test[:pred_steps], pred))

    # Megfelelő hosszra vágott pályák
    real_x = x[train_size:train_size + pred_steps]
    real_y = y[train_size:train_size + pred_steps]
    real_z = z[train_size:train_size + pred_steps]

    pred_x = pred
    pred_y = y[train_size:train_size + pred_steps]
    pred_z = z[train_size:train_size + pred_steps]

    # 💬 RMSE szétválasztva
    st.markdown(
        f"<div style='margin-top: 20px; font-size:18px; font-weight:bold;'>"
        f"📉 RMSE: <span style='color:#004d99'>{rmse:.4f}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # 🎯 3D vizualizáció Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=real_x, y=real_y, z=real_z,
                               mode='lines', name='Valós pálya', line=dict(color='blue')))
    fig.add_trace(go.Scatter3d(x=pred_x, y=pred_y, z=pred_z,
                               mode='lines', name='Predikció', line=dict(color='red')))

    fig.update_layout(
        scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='z'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
