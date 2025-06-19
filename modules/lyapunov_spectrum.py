# modules/lyapunov_spectrum.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def generate_lorenz_trajectory(dt=0.01, steps=10000, sigma=10, rho=28, beta=8/3):
    def lorenz(x, y, z, sigma, rho, beta):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return dx, dy, dz

    xs = np.empty(steps)
    ys = np.empty(steps)
    zs = np.empty(steps)
    xs[0], ys[0], zs[0] = 1.0, 1.0, 1.0

    for i in range(1, steps):
        dx, dy, dz = lorenz(xs[i - 1], ys[i - 1], zs[i - 1], sigma, rho, beta)
        xs[i] = xs[i - 1] + dx * dt
        ys[i] = ys[i - 1] + dy * dt
        zs[i] = zs[i - 1] + dz * dt

    return np.vstack((xs, ys, zs))

def calculate_lyapunov(trajectory):
    n = trajectory.shape[1]
    eps = 1e-6
    d0 = np.linalg.norm(trajectory[:, 1] - trajectory[:, 0]) + eps
    lyapunov_sum = 0

    for i in range(1, n - 1):
        d = np.linalg.norm(trajectory[:, i + 1] - trajectory[:, i])
        lyapunov_sum += np.log(abs(d / d0))

    return lyapunov_sum / (n - 2)

def run():
    st.title("🌌 Lyapunov Spektrum Vizualizáció")
    st.markdown("Ez a modul a Lorenz-rendszer pályáiból számítja a legnagyobb Lyapunov-exponenst.")

    steps = st.slider("Iterációk száma", 1000, 20000, 10000, step=1000)
    dt = st.number_input("Időlépés (dt)", 0.001, 0.1, 0.01, step=0.001)

    with st.spinner("Lorenz-rendszer generálása..."):
        traj = generate_lorenz_trajectory(dt=dt, steps=steps)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(traj[0], traj[1], traj[2], lw=0.5)
    ax.set_title("Lorenz-attractor")
    st.pyplot(fig)

    with st.spinner("Lyapunov-exponens számítása..."):
        lyap = calculate_lyapunov(traj)

    st.success(f"Legnagyobb Lyapunov-exponens: {lyap:.5f}")
