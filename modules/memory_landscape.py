import streamlit as st
import numpy as np
import plotly.graph_objects as go

def run():
    st.subheader("🧠 Memória tájkép (Pro) – Domborzati térkép")

    delay_range = st.slider("Késleltetés tartomány", 1, 20, (2, 10))
    noise_range = st.slider("Zaj szórás tartomány", 0.0, 2.0, (0.1, 1.0))

    delay_values = np.arange(delay_range[0], delay_range[1] + 1)
    noise_values = np.linspace(noise_range[0], noise_range[1], 20)
    delay_grid, noise_grid = np.meshgrid(delay_values, noise_values)

    # Mesterséges memória modell – szinuszos viselkedéssel és exponenciális zajhatással
    memory_capacity = np.exp(-noise_grid) * np.sin(delay_grid / 2) + 0.1 * np.random.rand(*delay_grid.shape)

    fig = go.Figure(data=[go.Surface(
        z=memory_capacity,
        x=delay_grid,
        y=noise_grid,
        colorscale='Viridis'
    )])

    fig.update_layout(
        title="🗺️ Memória domborzat",
        scene=dict(
            xaxis_title='Késleltetés',
            yaxis_title='Zaj szórás',
            zaxis_title='Memória kapacitás'
        ),
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)
