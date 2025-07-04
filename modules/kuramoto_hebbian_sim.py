import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# 🧮 Kuramoto–Hebbian szimuláció
def simulate_kuramoto_hebbian(N=10, K=1.0, eta=0.01, T=10, dt=0.1):
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(1.0, 0.1, N)
    W = np.ones((N, N)) - np.eye(N)

    time_points = int(T / dt)
    sync = []
    theta_history = []
    W_history = []

    for _ in range(time_points):
        theta_diff = theta[:, None] - theta[None, :]
        dtheta = omega + (K / N) * np.sum(W * np.sin(-theta_diff), axis=1)
        theta += dtheta * dt
        W += eta * np.cos(theta_diff) * dt
        np.fill_diagonal(W, 0)

        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        sync.append(r)
        theta_history.append(theta.copy())
        W_history.append(W.copy())

    return np.array(sync), np.array(theta_history), np.array(W_history)

# 🚀 Streamlit alkalmazás
def run():
    st.set_page_config(layout="wide")
    st.title("🧠 Kuramoto–Hebbian háló – Tanuló oszcillátorok dinamikája")

    st.markdown("""
    A **Kuramoto–Hebbian modell** egy dinamikus rendszer, ahol az oszcillátorok nem csak szinkronizálódnak, 
    hanem **tanulnak is a kapcsolatok erősségein keresztül** – hasonlóan a Hebbian-elvhez: *"Ami együtt sül, együtt kapcsolódik."*
    """)

    # 🎛️ Paraméterek
    st.sidebar.header("⚙️ Szimulációs paraméterek")
    N = st.sidebar.slider("🧩 Oszcillátorok száma (N)", 5, 50, 10)
    K = st.sidebar.slider("📡 Kapcsolási erősség (K)", 0.0, 5.0, 1.0, 0.1)
    eta = st.sidebar.slider("🧠 Tanulási ráta (η)", 0.001, 0.1, 0.01, 0.001)
    T = st.sidebar.slider("⏱️ Szimuláció ideje (T)", 5, 50, 10)
    dt = 0.05
    palette = st.sidebar.selectbox("🎨 Színséma (3D)", ["Viridis", "Turbo", "Electric", "Plasma", "Rainbow"])

    # ▶️ Szimuláció futtatása
    sync, theta_hist, W_hist = simulate_kuramoto_hebbian(N=N, K=K, eta=eta, T=T, dt=dt)

    # 📈 Szinkronizációs index (2D)
    st.subheader("📊 Szinkronizáció mértéke az idő függvényében")
    fig1, ax1 = plt.subplots()
    ax1.plot(sync)
    ax1.set_xlabel("Időlépések")
    ax1.set_ylabel("Szinkronizáció (R)")
    ax1.set_title("Kuramoto–Hebbian szinkronizáció")
    st.pyplot(fig1)

    # 🌐 3D vizualizáció a végső állapotra
    st.subheader("🌐 3D hálózat vizualizáció – Oszcillátor fázis színezéssel")

    final_theta = theta_hist[-1]
    pos = np.array([
        [np.cos(2 * np.pi * i / N), np.sin(2 * np.pi * i / N), 0]
        for i in range(N)
    ])

    edge_x, edge_y, edge_z = [], [], []
    W_final = W_hist[-1]

    for i in range(N):
        for j in range(i + 1, N):
            if W_final[i, j] > 0.05:  # csak erősebb kapcsolatok
                xi, yi, zi = pos[i]
                xj, yj, zj = pos[j]
                edge_x += [xi, xj, None]
                edge_y += [yi, yj, None]
                edge_z += [zi, zj, None]

    fig3d = go.Figure()

    fig3d.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='lightgray', width=1),
        opacity=0.5,
        name="Kapcsolatok"
    ))

    fig3d.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=final_theta,
            colorscale=palette,
            line=dict(width=0.5, color='black')
        ),
        name="Oszcillátorok"
    ))

    fig3d.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False)
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig3d, use_container_width=True)

    # 📘 Tudományos háttér
    st.markdown("### 📘 Tudományos háttér – Dinamika és tanulás")

    st.latex(r"""
    \frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^N W_{ij} \sin(\theta_j - \theta_i)
    """)

    st.latex(r"""
    \frac{dW_{ij}}{dt} = \eta \cos(\theta_i - \theta_j)
    """)

    st.markdown("""
    - **$\theta_i$**: az *i*-edik oszcillátor fázisa  
    - **$\omega_i$**: sajátfrekvencia (általában normál eloszlásból)  
    - **$W_{ij}$**: dinamikusan tanuló kapcsolaterősség  
    - **$K$**: globális kapcsolási állandó  
    - **$\eta$**: tanulási ráta  
    - A Hebbian tanulás lényege, hogy a kapcsolat erősödik, ha a fáziskülönbség kicsi (közel szinkron)

    A szinkronizáció mértékét az ún. **order parameter** mutatja:
    """)

    st.latex(r"""
    R(t) = \left| \frac{1}{N} \sum_{j=1}^N e^{i\theta_j(t)} \right|
    """)

    # 📈 Kapcsolaterősségek átlaga időben
    avg_weights = np.mean(W_hist, axis=(1, 2))

    st.subheader("🔗 Átlagos kapcsolaterősség időbeli változása")
    fig2, ax2 = plt.subplots()
    ax2.plot(avg_weights)
    ax2.set_xlabel("Időlépések")
    ax2.set_ylabel("Átlagos $W_{ij}$")
    ax2.set_title("Hebbian tanulás hatása")
    st.pyplot(fig2)

    # 📝 Jegyzetek
    st.subheader("📝 Megfigyelések és jegyzetek")
    st.text_area("Mit tapasztaltál a szinkronizáció és tanulás során?", placeholder="Írd ide...")

# ✅ ReflectAI-kompatibilis export
app = run
