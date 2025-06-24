import streamlit as st import numpy as np import matplotlib.pyplot as plt

def simulate_kuramoto_with_noise(N=100, K=1.0, noise_strength=0.1, T=10, dt=0.01): time = np.arange(0, T, dt) theta = np.random.uniform(0, 2 * np.pi, N) omega = np.random.normal(0, 1, N) sync_history = []

for t in time:
    noise = noise_strength * np.random.normal(0, 1, N)
    coupling = (K / N) * np.sum(np.sin(np.subtract.outer(theta, theta)), axis=1)
    dtheta = omega + coupling + noise
    theta += dtheta * dt
    order_param = np.abs(np.mean(np.exp(1j * theta)))
    sync_history.append(order_param)

return time, sync_history

def run(): st.header("🔊 Zajtűrés és szinkronizációs robusztusság")

N = st.slider("Oszcillátorok száma (N)", 10, 300, 100)
K = st.slider("Kapcsolási erősség (K)", 0.0, 5.0, 1.0)
noise_strength = st.slider("Zaj erőssége", 0.0, 1.0, 0.1)
T = st.slider("Szimuláció ideje", 1, 50, 10)
dt = 0.01

time, sync_history = simulate_kuramoto_with_noise(N, K, noise_strength, T, dt)

fig, ax = plt.subplots()
ax.plot(time, sync_history, label="Szinkronizációs fok (r)")
ax.set_xlabel("Idő")
ax.set_ylabel("Szinkronizációs fok")
ax.set_title("Zaj hatása a szinkronizációra")
ax.grid(True)
st.pyplot(fig)

