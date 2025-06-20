import streamlit as st
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

def lorenz_system(x, y, z, s=10, r=28, b=2.667):
    dx = s * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z
    return dx, dy, dz

def generate_lorenz_data(dt, steps, x0=0., y0=1., z0=1.05):
    xs, ys, zs = [x0], [y0], [z0]
    for _ in range(steps - 1):
        dx, dy, dz = lorenz_system(xs[-1], ys[-1], zs[-1])
        xs.append(xs[-1] + dx * dt)
        ys.append(ys[-1] + dy * dt)
        zs.append(zs[-1] + dz * dt)
    return np.array(xs), np.array(ys), np.array(zs)

def embed_time_series(data, delay, dimension):
    N = len(data)
    M = N - (dimension - 1) * delay
    if M <= 0:
        return np.empty((0, dimension))
    return np.array([data[i:i + delay * dimension:delay] for i in range(M)])

def run():
    st.subheader("🧠 Lorenz előrejelzés – Modellválasztással")

    # Interaktív modelválasztó
    model_type = st.selectbox("🤖 Válassz prediktív modellt", ["Ridge", "Lasso", "MLP"])

    dt = st.slider("🕒 Időlépés", 0.005, 0.05, 0.01)
    steps = st.slider("🔁 Időlépések száma", 1000, 3000, 1500)
    delay = st.slider("⏱️ Késleltetés", 1, 20, 3)
    dimension = st.slider("📐 Beágyazás dimenziója", 2, 10, 5)
    pred_steps = st.slider("🎯 Előrejelzendő lépések száma", 1, 50, 20)

    # Lorenz pálya
    x, y, z = generate_lorenz_data(dt, steps)
    X_embed = embed_time_series(x, delay, dimension)
    y_target = x[(dimension - 1) * delay + pred_steps:]

    min_len = min(len(X_embed), len(y_target))
    X_embed = X_embed[:min_len]
    y_target = y_target[:min_len]

    valid_mask = np.all(np.isfinite(X_embed), axis=1) & np.isfinite(y_target)
    X_data = X_embed[valid_mask]
    y_data = y_target[valid_mask]

    if len(X_data) == 0:
        st.error("❌ Nincs elég érvényes adat. Próbálj más paramétereket.")
        return

    # Modell kiválasztása
    if model_type == "Ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "Lasso":
        model = Lasso(alpha=0.01, max_iter=10000)
    elif model_type == "MLP":
        model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=500)

    model.fit(X_data, y_data)
    pred_x = model.predict(X_data)
    pred_x_full = np.concatenate([X_data[:, 0], pred_x])[:len(x)]

    # 3D plot – valós és predikált pályák
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Valós pálya', line=dict(color='blue')))
    fig.add_trace(go.Scatter3d(x=pred_x_full, y=y, z=z, mode='lines', name='Predikció', line=dict(color='red')))

    fig.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    rmse = np.sqrt(mean_squared_error(x[:len(pred_x_full)], pred_x_full))
    st.markdown(f"### 📈 {model_type} predikció – RMSE: `{rmse:.4f}`")
    st.plotly_chart(fig, use_container_width=True)
