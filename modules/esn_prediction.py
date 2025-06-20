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
        embedding_delay = st.slider("⏱️ Beágyazás késleltetése", 1, 10, 3)
        dim = st.slider("📐 Beágyazási dimenzió", 2, 10, 5)

    # Szimuláció paraméterei
    t_max = 50
    dt = 0.01
    t = np.arange(0, t_max, dt)
    initial_state = [1.0, 1.0, 1.0]
    sol = solve_ivp(lorenz, [0, t_max], initial_state, t_eval=t, method="RK45")
    x, y, z = sol.y

    # Embedder
    def create_embedding(data, delay, dim):
        N = len(data) - delay * (dim - 1)
        if N <= 0:
            return np.empty((0, dim))
        return np.array([data[i:i + delay * dim:delay] for i in range(N)])

    X_embed = create_embedding(x, embedding_delay, dim)
    y_target = x[embedding_delay * dim:embedding_delay * dim + len(X_embed)]
    valid_mask = np.all(np.isfinite(X_embed), axis=1) & np.isfinite(y_target)
    X_embed = X_embed[valid_mask]
    y_target = y_target[valid_mask]

    # Modell
    model = Ridge(alpha=1.0)
    model.fit(X_embed[:-steps], y_target[:-steps])
    y_pred = model.predict(X_embed[-steps:])
    pred_x = np.concatenate([x[:-(steps)], y_pred])
    rmse = np.sqrt(mean_squared_error(x[-steps:], y_pred))

    # 📊 Eredmények szekció
    st.markdown("---")
    st.markdown("### 📈 Eredmények")
    st.markdown(f"**RMSE (Root Mean Squared Error):** `{rmse:.4f}`")

    # 💡 Info kártya
    with st.expander("ℹ️ Modellről bővebben"):
        st.write(
            "Ez a szimuláció Echo State Network-öt (ESN) használ a Lorenz attraktor x-komponensének predikciójára. "
            "A bemeneteket késleltetett koordinátákból állítjuk elő (delay embedding)."
        )

    # 📦 Konténer a Plotly ábrához
    st.markdown("---")
    st.markdown("### 🎯 3D Lorenz pálya – Valós vs. Predikció")

    with st.container():
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Valós pálya', line=dict(color='blue')))
        fig.add_trace(go.Scatter3d(
            x=pred_x[-steps:],
            y=y[-steps:],
            z=z[-steps:],
            mode='lines',
            name='Predikció',
            line=dict(color='red')
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=600,
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)
