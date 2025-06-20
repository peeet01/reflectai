import streamlit as st
import numpy as np
from sklearn.linear_model import Ridge
import plotly.graph_objects as go

def run():
    st.subheader("🧠 ESN Predikció")

    delay = st.slider("Késleltetés (delay)", 2, 10, 5)
    dim = st.slider("Beágyazási dimenzió", 3, 10, 5)
    pred_steps = st.slider("Előrejelzendő lépések száma", 1, 50, 20)

    def lorenz(x, y, z, s=10, r=28, b=2.667):
        dx = s * (y - x)
        dy = r * x - y - x * z
        dz = x * y - b * z
        return dx, dy, dz

    dt = 0.01
    num_steps = 2000
    xyz = np.empty((num_steps, 3))
    xyz[0] = (0., 1., 1.05)

    for i in range(1, num_steps):
        dx, dy, dz = lorenz(*xyz[i - 1])
        xyz[i] = xyz[i - 1] + dt * np.array([dx, dy, dz])

    xyz -= xyz.mean(axis=0)
    xyz /= xyz.std(axis=0)

    def time_delay_embed(data, delay, dim):
        N = len(data)
        M = N - (dim - 1) * delay
        if M <= 0:
            return np.empty((0, dim))
        return np.array([data[i:i + delay * dim:delay] for i in range(M)])

    x = xyz[:, 0]
    X_embed = time_delay_embed(x, delay, dim)
    y_target = x[(dim - 1) * delay + pred_steps:]

    min_len = min(len(X_embed), len(y_target))
    X_data = X_embed[:min_len]
    y_data = y_target[:min_len]

    valid_mask = np.all(np.isfinite(X_data), axis=1) & np.isfinite(y_data)
    X_data = X_data[valid_mask]
    y_data = y_data[valid_mask]

    if len(X_data) < 10:
        st.warning("⚠️ Túl kevés érvényes adat. Próbálj más beállítást.")
        return

    model = Ridge(alpha=1.0)
    model.fit(X_data, y_data)
    y_pred = model.predict(X_data)

    rmse = np.sqrt(np.mean((y_pred - y_data) ** 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xyz[:len(y_pred), 0],
        y=xyz[:len(y_pred), 1],
        z=xyz[:len(y_pred), 2],
        mode='lines',
        line=dict(color='blue'),
        name='Valós pálya'
    ))
    fig.add_trace(go.Scatter3d(
        x=xyz[:len(y_pred), 0],
        y=xyz[:len(y_pred), 1],
        z=y_pred,
        mode='lines',
        line=dict(color='red'),
        name='Predikció'
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z / pred'
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
    )

    st.markdown("### 🎯 RMSE: `{:.4f}`".format(rmse))
    st.plotly_chart(fig, use_container_width=True)
