import streamlit as st
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

def generate_lorenz(dt, steps, sigma=10, rho=28, beta=8/3):
    xyz = np.empty((steps, 3))
    xyz[0] = [1.0, 1.0, 1.0]
    for i in range(1, steps):
        x, y, z = xyz[i - 1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        xyz[i] = xyz[i - 1] + dt * np.array([dx, dy, dz])
    return xyz

def time_delay_embedding(data, delay, dimension):
    N = len(data) - delay * (dimension - 1)
    if N <= 0:
        return np.empty((0, dimension))
    return np.array([data[i:i + delay * dimension:delay] for i in range(N)])

def run():
    st.subheader("🌀 ESN predikció Lorenz attraktoron")

    # Paraméterek
    dt = st.slider("Időlépés (dt)", 0.001, 0.05, 0.01)
    steps = st.slider("Időlépések száma", 500, 3000, 1500)
    delay = st.slider("Késleltetés (delay)", 1, 20, 3)
    dim = st.slider("Beágyazás dimenziója", 2, 10, 5)
    predict_horizon = st.slider("Előrejelzendő lépések száma", 1, 50, 20)

    # Lorenz generálás
    data = generate_lorenz(dt, steps)
    x_series = data[:, 0]

    # Embed
    X_embed = time_delay_embedding(x_series, delay, dim)
    y_target = x_series[delay * dim : delay * dim + len(X_embed)]

    # Validáció
    valid_mask = np.all(np.isfinite(X_embed), axis=1) & np.isfinite(y_target)
    X_data = X_embed[valid_mask]
    y_data = y_target[valid_mask]

    # Modell tanítás
    if len(X_data) < predict_horizon:
        st.error("⚠️ Túl kevés adat az előrejelzéshez!")
        return

    model = Ridge(alpha=1.0)
    model.fit(X_data[:-predict_horizon], y_data[predict_horizon:])

    # Előrejelzés
    preds = model.predict(X_data[:-predict_horizon])
    true_vals = y_data[predict_horizon:]

    # RMSE számítás
    rmse = mean_squared_error(true_vals, preds, squared=False)

    # Vizuális elrendezés: szöveg külön, ábra alá
    with st.container():
        st.markdown("### 🌪️ ESN Predikció")
        st.markdown(f"📉 **RMSE:** `{rmse:.4f}`")
        st.markdown("---")

    # 3D vizualizáció
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2],
                                mode='lines', name='Valós pálya',
                                line=dict(color='blue')))
    pred_lorenz = np.zeros((len(preds), 3))
    pred_lorenz[:, 0] = preds
    pred_lorenz[:, 1:] = data[:len(preds), 1:]  # y,z csak díszítésként

    fig.add_trace(go.Scatter3d(x=pred_lorenz[:, 0],
                               y=pred_lorenz[:, 1],
                               z=pred_lorenz[:, 2],
                               mode='lines',
                               name='Predikció',
                               line=dict(color='red')))
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
