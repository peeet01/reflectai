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


def neuron_network_3d(theta, radius=5):
    N = len(theta)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros(N)

    phase_colors = [f"hsl({int(np.degrees(t) % 360)}, 100%, 50%)" for t in theta]
    fig = go.Figure()

    for i in range(N):
        fig.add_trace(go.Scatter3d(
            x=[x[i], x[i] + 0.4 * np.cos(theta[i])],
            y=[y[i], y[i] + 0.4 * np.sin(theta[i])],
            z=[z[i], z[i] + 0.5 * np.sin(theta[i])],
            mode='lines+markers',
            marker=dict(size=4, color=phase_colors[i]),
            line=dict(color=phase_colors[i], width=2),
            showlegend=False
        ))

    for i in range(N):
        for j in range(i + 1, N):
            fig.add_trace(go.Scatter3d(
                x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                mode='lines',
                line=dict(color='lightgray', width=0.5),
                hoverinfo="skip",
                showlegend=False
            ))

    fig.update_layout(
        title="🧠 3D Neuronháló Kuramoto-fázisokkal",
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def run(K=2.0, N=10):
    st.subheader("🧭 Kuramoto szinkronizáció szimuláció")

    # Paraméterek
    T = 200
    dt = 0.05

    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)
    initial_theta = theta.copy()

    r_values = []
    theta_std = []

    progress = st.progress(0)
    progress_text = st.empty()

    for t in range(T):
        theta = kuramoto_step(theta, omega, K, dt)
        r_values.append(compute_order_parameter(theta))
        theta_std.append(np.std(theta))

        # 🟣 Progress bar frissítése
        if t % max(1, T // 100) == 0 or t == T - 1:
            progress.progress((t + 1) / T)
            progress_text.text(f"⏳ Szimuláció folyamatban... {int((t+1)/T*100)}%")

    # 📊 Matplotlib ábrák
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

    # 🧠 3D neuronháló megjelenítés
    st.markdown("### 🌐 Térbeli neuronháló (fázisszinkronizáció színekkel)")
    fig3d = neuron_network_3d(theta)
    st.plotly_chart(fig3d, use_container_width=True)
