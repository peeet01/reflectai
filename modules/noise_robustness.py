import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def kuramoto_step(theta, omega, K, dt):
    N = len(theta)
    theta_matrix = np.subtract.outer(theta, theta)
    coupling = np.sum(np.sin(theta_matrix), axis=1)
    theta += (omega + (K / N) * coupling) * dt
    return theta

def compute_order_parameter(theta):
    return np.abs(np.sum(np.exp(1j * theta)) / len(theta))

def simulate_noise_robustness(N, T, dt, K_vals, noise_levels):
    results = np.zeros((len(K_vals), len(noise_levels)))

    for i, K in enumerate(K_vals):
        for j, noise in enumerate(noise_levels):
            theta = np.random.uniform(0, 2 * np.pi, N)
            omega = np.random.normal(0, 1, N)

            for _ in range(T):
                theta = kuramoto_step(theta, omega, K, dt)
                theta += np.random.normal(0, noise, N)  # Zaj hozzáadása

            order_param = compute_order_parameter(theta)
            results[i, j] = order_param
    return results

def run():
    st.subheader("🧪 Zajtűrés és szinkronizációs robusztusság (Pro)")

    N = st.slider("Oszcillátorok száma (N)", 10, 100, 30)
    T = st.slider("Iterációk száma (T)", 100, 1000, 500)
    dt = st.slider("Időlépés (dt)", 0.01, 0.1, 0.05)

    k_min, k_max = st.slider("Kapcsolási erősség tartománya (K)", 0.0, 10.0, (1.0, 5.0))
    noise_min, noise_max = st.slider("Zaj tartomány (szórás)", 0.0, 2.0, (0.0, 1.0))

    k_values = np.linspace(k_min, k_max, 30)
    noise_levels = np.linspace(noise_min, noise_max, 30)

    st.markdown("⏳ Szimuláció fut...")
    data = simulate_noise_robustness(N, T, dt, k_values, noise_levels)

    # 2D hőtérkép
    fig, ax = plt.subplots()
    c = ax.imshow(data, extent=[noise_min, noise_max, k_min, k_max], aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(c, ax=ax, label="Szinkronizációs index")
    ax.set_xlabel("Zaj szint (σ)")
    ax.set_ylabel("Kapcsolási erősség (K)")
    ax.set_title("🗺️ Zajtűrési hőtérkép")
    st.pyplot(fig)

    # 3D plot
    K_mesh, Noise_mesh = np.meshgrid(noise_levels, k_values)
    fig3d = go.Figure(data=[go.Surface(z=data, x=Noise_mesh, y=K_mesh, colorscale='Viridis')])
    fig3d.update_layout(title="🌌 3D Zajtűrési térkép",
                        scene=dict(xaxis_title='Zaj (σ)',
                                   yaxis_title='Kapcsolás (K)',
                                   zaxis_title='Szinkronizáció'),
                        margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig3d, use_container_width=True)
