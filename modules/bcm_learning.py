import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# BCM tanulási szabály több neuronra
def bcm_learning_multi(N=10, eta=0.01, tau=100, steps=500):
    w = np.random.rand(N, N) * 0.5
    theta = np.ones(N) * 0.1
    w_hist = np.zeros((steps, N, N))
    theta_hist = np.zeros((steps, N))
    y_hist = np.zeros((steps, N))
    
    x_input = np.random.randn(steps, N)

    for t in range(steps):
        x = x_input[t]
        y = w @ x
        for i in range(N):
            for j in range(N):
                dw = eta * x[j] * y[i] * (y[i] - theta[i])
                w[i, j] += dw
        theta += (y**2 - theta) / tau
        w_hist[t] = w
        theta_hist[t] = theta
        y_hist[t] = y

    return w_hist, theta_hist, y_hist, x_input

# 3D neuronháló megjelenítés – élvastagság és szín a súlyhoz skálázva
def draw_3d_network_dynamic(w_matrix):
    N = w_matrix.shape[0]
    pos = np.array([
        [np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N), 0.5 * np.sin(4 * np.pi * i / N)]
        for i in range(N)
    ])

    fig = go.Figure()

    max_weight = np.max(np.abs(w_matrix)) or 1.0

    for i in range(N):
        for j in range(N):
            if i != j:
                weight = abs(w_matrix[i, j])
                norm_weight = weight / max_weight
                fig.add_trace(go.Scatter3d(
                    x=[pos[i, 0], pos[j, 0]],
                    y=[pos[i, 1], pos[j, 1]],
                    z=[pos[i, 2], pos[j, 2]],
                    mode='lines',
                    line=dict(
                        color=f'rgba({255*norm_weight:.0f}, 50, 255, {0.2 + 0.6 * norm_weight:.2f})',
                        width=1 + 4 * norm_weight
                    ),
                    showlegend=False
                ))

    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(size=8, color='orange'),
        name='Neuronok'
    ))

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )

    return fig

# Streamlit futtatás
def run():
    st.title("🧠 BCM Learning – Több neuron 3D vizualizációval")

    steps = st.slider("Szimuláció lépései", 100, 2000, 500, step=100)
    eta = st.slider("Tanulási ráta (η)", 0.001, 0.1, 0.01, step=0.001)
    tau = st.slider("Küszöb időállandó (τ)", 10, 500, 100, step=10)
    N = st.slider("Neuronok száma", 5, 20, 10)

    w_hist, theta_hist, y_hist, x_input = bcm_learning_multi(N=N, eta=eta, tau=tau, steps=steps)
    selected_step = st.slider("Vizualizált lépés", 0, steps - 1, steps - 1)
    current_w = w_hist[selected_step]

    st.subheader("📈 Tanulási dinamika példaneuronra (neuron 0)")
    fig, ax = plt.subplots()
    ax.plot(theta_hist[:, 0], label="Küszöb (θ₀)")
    ax.plot(y_hist[:, 0], label="Kimenet (y₀)")
    ax.set_title("Időbeli változás")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔬 3D neuronháló vizualizáció – súly megjelenítéssel")
    st.plotly_chart(draw_3d_network_dynamic(current_w))

    st.subheader("📥 CSV letöltés")
    df = pd.DataFrame(w_hist.reshape(steps, -1))
    st.download_button("Letöltés CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="bcm_learning.csv")

    st.markdown("""
### 📚 Tudományos háttér

A **BCM-szabály** a szinaptikus plaszticitás egyik biológiailag megalapozott modellje, amely egy **nemlineáris aktivitásfüggő** tanulási küszöböt használ.

**Formális leírás:**

- Súlyváltozás:  
  \( \frac{dw}{dt} = \eta \cdot x \cdot y \cdot (y - \theta) \)

- Küszöbszint változása:  
  \( \frac{d\theta}{dt} = \frac{1}{\tau} (y^2 - \theta) \)

**Jelentőség:**

- Homeosztatikus stabilitás  
- Szelektív tanulás  
- Szenzoros rendszer fejlődésének modellezése  
    """)

# Kötelező hozzárendelés
app = run
