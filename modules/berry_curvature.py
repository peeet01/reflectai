import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Berry-görbület számítása ---
def compute_berry_curvature(kx, ky):
    delta = 0.1
    d = np.array([
        np.sin(kx),
        np.sin(ky),
        delta + np.cos(kx) + np.cos(ky)
    ])
    norm = np.linalg.norm(d)
    d_hat = d / norm
    return 0.5 * d_hat[2] / (norm**2 + 1e-8)

# --- Berry-fázis számítása kör mentén a k-térben ---
def compute_berry_phase(radius=1.0, center=(0.0, 0.0), num_points=200):
    delta = 0.1
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    phase_sum = 0.0

    for i in range(num_points):
        theta1 = thetas[i]
        theta2 = thetas[(i + 1) % num_points]
        k1 = np.array([
            center[0] + radius * np.cos(theta1),
            center[1] + radius * np.sin(theta1)
        ])
        k2 = np.array([
            center[0] + radius * np.cos(theta2),
            center[1] + radius * np.sin(theta2)
        ])

        d1 = np.array([np.sin(k1[0]), np.sin(k1[1]), delta + np.cos(k1[0]) + np.cos(k1[1])])
        d2 = np.array([np.sin(k2[0]), np.sin(k2[1]), delta + np.cos(k2[0]) + np.cos(k2[1])])
        u1 = d1 / np.linalg.norm(d1)
        u2 = d2 / np.linalg.norm(d2)

        inner_prod = np.vdot(u1, u2)
        phase = np.angle(inner_prod)
        phase_sum += phase

    return np.real(phase_sum)

# --- 3D vizualizáció a d-vektor pályájáról ---
def plot_berry_phase_3d(radius=1.0, center=(0.0, 0.0), num_points=200):
    delta = 0.1
    thetas = np.linspace(0, 2 * np.pi, num_points)
    traj = []

    for theta in thetas:
        kx = center[0] + radius * np.cos(theta)
        ky = center[1] + radius * np.sin(theta)
        d = np.array([
            np.sin(kx),
            np.sin(ky),
            delta + np.cos(kx) + np.cos(ky)
        ])
        d /= np.linalg.norm(d)
        traj.append(d)

    traj = np.array(traj)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="d-vektor pálya", color="purple")
    ax.set_title("3D Berry-fázis vizualizáció – d-vektor pálya")
    ax.set_xlabel("d₁")
    ax.set_ylabel("d₂")
    ax.set_zlabel("d₃")
    ax.legend()
    st.pyplot(fig)

# --- Streamlit fő modul ---
def run():
    st.header("🌀 Topológiai védettség és Berry-görbület")
    st.markdown("Ez a szimuláció a 2D Brillouin-zónában vizsgálja a Berry-görbület eloszlását.")

    # --- Berry-görbület térkép ---
    N = st.slider("Pontok száma tengelyenként", 30, 150, 80, 10)
    kx_vals = np.linspace(-np.pi, np.pi, N)
    ky_vals = np.linspace(-np.pi, np.pi, N)
    curvature = np.zeros((N, N))

    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            curvature[j, i] = compute_berry_curvature(kx, ky)

    fig, ax = plt.subplots()
    c = ax.contourf(kx_vals, ky_vals, curvature, levels=50, cmap='coolwarm')
    fig.colorbar(c, ax=ax, label="Berry-görbület")
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    ax.set_title("Berry-görbület a Brillouin-zónában")
    st.pyplot(fig)

    # --- ÚJ: Berry-fázis kiszámítása kör mentén ---
    with st.expander("🔄 Berry-fázis számítása (körintegrál)"):
        st.markdown("A Berry-fázis egy kvantummechanikai geometriai fázis, amit egy zárt kört leírva a k-térben számítunk.")
        radius = st.slider("Kör sugara (k-tér)", 0.1, 3.0, 1.0, 0.1)
        center_kx = st.slider("Kör középpontja – kx", -np.pi, np.pi, 0.0)
        center_ky = st.slider("Kör középpontja – ky", -np.pi, np.pi, 0.0)
        num_points = st.slider("Pontok száma a kör mentén", 50, 500, 200, 10)

        berry_phase = compute_berry_phase(radius=radius, center=(center_kx, center_ky), num_points=num_points)
        st.success(f"📐 Berry-fázis értéke ≈ `{berry_phase:.4f}` radián")

    # --- ÚJ: 3D vizualizáció ---
    with st.expander("🌐 3D Berry-fázis pálya"):
        st.markdown("A kvantumállapot irányának változása 3D térben (d-vektor normalizált pályája).")
        plot_berry_phase_3d(radius=radius, center=(center_kx, center_ky), num_points=num_points)

# 🔧 Kötelező dinamikus modul belépési pont
app = run
