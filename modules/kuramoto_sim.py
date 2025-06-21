import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go


def kuramoto_step(theta, omega, K, dt):
    N = len(theta)
    theta_diff = np.subtract.outer(theta, theta)
    coupling = np.sum(np.sin(theta_diff), axis=1)
    return theta + dt * (omega + (K / N) * coupling)


def compute_order_parameter(theta):
    return np.abs(np.sum(np.exp(1j * theta)) / len(theta))


def plot_3d_neurons(theta):
    N = len(theta)
    radius = 5
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(N)

    colors = [f"hsl({int((t % (2 * np.pi)) / (2 * np.pi) * 360)},100%,50%)" for t in theta]

    fig = go.Figure()

    # Neuron testek
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=6, color=colors),
        text=[f'Neuron {i+1}' for i in range(N)],
        name='Neuronok'
    ))

    # Dendritszerű kiágazások
    for i in range(N):
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[i]],
            y=[y[i], y[i]],
            z=[z[i], z[i] + 1.5],
            mode='lines',
            line=dict(color=colors[i], width=3),
            showlegend=False
        ))

    fig.update_layout(
        title='🧠 3D neuron háló szinkronizáció vizualizáció',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z (dendrit)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    st.plotly_chart(fig, use_container_width=True)


def run(K=2.0, N=10):
    st.subheader("🧭 Kuramoto szinkronizáció szimuláció")

    T = 200
    dt = 0.05

    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)
    initial_theta = theta.copy()

    r_values = []
    theta_std = []

    for _ in range(T):
        theta = kuramoto_step(theta, omega, K, dt)
        r_values.append(compute_order_parameter(theta))
        theta_std.append(np.std(theta))

    # Ábrák: Kezdeti–Végső fázis + r(t) + szórás
    fig1, ax1 = plt.subplots(subplot_kw=dict(polar=True))
    ax1.set_title("🔵 Kezdeti fáziseloszlás")
    ax1.scatter(initial_theta, np.ones(N), c='blue', alpha=0.75)

    fig2, ax2 = plt.subplots(subplot_kw=dict(polar=True))
    ax2.set_title("🔴 Végső fáziseloszlás")
    ax2.scatter(theta, np.ones(N), c='red', alpha=0.75)

    fig3, ax3 = plt.subplots()
    ax3.plot(r_values, label="Szinkronizációs index r(t)", color='purple')
    ax3.set_xlabel("Időlépések")
    ax3.set_ylabel("r érték")
    ax3.set_title("📈 Szinkronizációs index időfüggése")
    ax3.grid(True)
    ax3.legend()

    fig4, ax4 = plt.subplots()
    ax4.plot(theta_std, label="Fázis szórása", color='green')
    ax4.set_xlabel("Időlépések")
    ax4.set_ylabel("Szórás")
    ax4.set_title("📉 Fáziseloszlás szórása időben")
    ax4.grid(True)
    ax4.legend()

    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.pyplot(fig4)

    st.markdown(f"**🔎 Végső szinkronizációs index:** `{r_values[-1]:.3f}`")
    st.markdown(f"**📊 Végső szórás:** `{theta_std[-1]:.3f}`")

    # Új: 3D neuronvizualizáció
    st.markdown("### 🧠 3D neuronháló vizualizáció (fázis szerinti színezés)")
    plot_3d_neurons(theta)
