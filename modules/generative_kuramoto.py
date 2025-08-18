import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# Segédfüggvények
# ----------------------------
def generate_graph(n_nodes: int, topo: str, p: float, k_ws: int = 4) -> nx.Graph:
    if topo == "Erdős–Rényi (ER)":
        return nx.erdos_renyi_graph(n_nodes, p, seed=42)
    elif topo == "Small-World (Watts–Strogatz)":
        k_ws = max(2, min(n_nodes - 1, k_ws | 1))  # páratlan->páratlan oké
        return nx.watts_strogatz_graph(n_nodes, k_ws, p, seed=42)
    else:  # Scale-Free (Barabási–Albert)
        m = max(1, int(max(1, round(p * n_nodes / 2))))
        m = min(m, n_nodes - 1)
        return nx.barabasi_albert_graph(n_nodes, m, seed=42)

def sample_omega(n: int, dist: str, loc=1.0, scale=0.1, bimodal_shift=0.8):
    if dist == "Normál(μ,σ)":
        return np.random.normal(loc, scale, n)
    if dist == "Egyenletes":
        return np.random.uniform(loc - np.sqrt(3)*scale, loc + np.sqrt(3)*scale, n)
    # Bimodális: két szimmetrikus csúcs
    half = n // 2
    w1 = np.random.normal(loc + bimodal_shift, scale, half)
    w2 = np.random.normal(loc - bimodal_shift, scale, n - half)
    return np.concatenate([w1, w2])

def hsv_phase_colors(theta):
    """Theta -> HSL stringek (Plotly barátságos)."""
    phase = (theta % (2*np.pi)) / (2*np.pi)  # [0,1)
    return [f"hsl({int(360*p)},100%,50%)" for p in phase]

def largest_eigenvalue(A):
    try:
        # valós, szimmetrikus esetben gyors
        vals = np.linalg.eigvals(A)
        return float(np.max(np.real(vals)))
    except Exception:
        return float(np.linalg.norm(A, 2))

def g0_from_sigma(sigma):
    # sűrűség 0-ban, normál eloszlás esetén
    if sigma <= 0:
        return np.inf
    return 1.0 / (sigma * np.sqrt(2*np.pi))

# ----------------------------
# Kuramoto-szimuláció (történettel)
# ----------------------------
def simulate_kuramoto(G, K, omega, t_max=10.0, dt=0.05, record=50):
    """
    Visszaadja: t, R(t) lista, theta_final, theta_history (frames x N)
    record: ennyi frame-et egyenletesen mintázunk ki a vizualizációhoz
    """
    N = len(G.nodes)
    A = nx.to_numpy_array(G)
    theta = np.random.uniform(0, 2*np.pi, N)
    t = np.arange(0, t_max, dt)
    order_hist = []
    frames_idx = np.linspace(0, len(t)-1, num=max(2, record), dtype=int)
    theta_frames = []

    for k, _ in enumerate(t):
        dtheta = omega + (K / N) * np.sum(A * np.sin(theta[:, None] - theta), axis=1)
        theta += dtheta * dt
        R = np.abs(np.sum(np.exp(1j * theta)) / N)
        order_hist.append(R)
        if k in set(frames_idx):
            theta_frames.append(theta.copy())

    if len(theta_frames) == 0:
        theta_frames = [theta.copy()]
    return t, np.array(order_hist), theta.copy(), np.array(theta_frames)

# ----------------------------
# 2D szinkronizációs plot
# ----------------------------
def plot_sync(t, R):
    fig, ax = plt.subplots()
    ax.plot(t, R)
    ax.set_xlabel("Idő")
    ax.set_ylabel("Szinkronizáció (R)")
    ax.set_title("Kuramoto szinkronizációs dinamika")
    ax.set_ylim(0, 1.05)
    st.pyplot(fig)

# ----------------------------
# 3D gráf
# ----------------------------
def plot_graph_3d(G, theta, title, edge_cap=900):
    pos = nx.spring_layout(G, dim=3, seed=42)
    node_xyz = np.array([pos[v] for v in G.nodes()])
    # élek ritkítása
    edges = np.array([(u, v) for u, v in G.edges()])
    if len(edges) > edge_cap:
        idx = np.random.choice(len(edges), edge_cap, replace=False)
        edges = edges[idx]

    edge_x, edge_y, edge_z = [], [], []
    for u, v in edges:
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(200,200,200,0.35)", width=1),
        hoverinfo="none",
        name="Élek",
        opacity=0.6,
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=node_xyz[:, 0], y=node_xyz[:, 1], z=node_xyz[:, 2],
        mode="markers",
        marker=dict(
            size=7,
            color=hsv_phase_colors(theta),  # fázis → szín
            line=dict(color="black", width=0.5)
        ),
        name="Oszcillátorok"
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Streamlit App
# ----------------------------
def run():
    st.set_page_config(layout="wide")
    st.title("🌐 Generatív Kuramoto hálózat")

    st.markdown(
        "Választható topológiák és sajátfrekvencia-eloszlások, "
        "kritikus kupling becslés **Kc**, valamint időbeli pillanatképek."
    )

    # ---- Beállítások
    with st.sidebar:
        st.header("⚙️ Beállítások")
        topo = st.selectbox("Topológia", ["Erdős–Rényi (ER)", "Small-World (Watts–Strogatz)", "Scale-Free (Barabási–Albert)"])
        n_nodes = st.slider("🧩 Csomópontok száma", 10, 200, 60, step=1)
        p = st.slider("🔗 Paraméter p (ER/WS) – vagy sűrűség jelleg", 0.01, 1.0, 0.15, step=0.01)
        ws_k = st.slider("WS: k (szomszédok)", 2, 20, 6, step=2)
        st.divider()
        omega_dist = st.selectbox("ω eloszlás", ["Normál(μ,σ)", "Egyenletes", "Bimodális (±Ω)"])
        mu = st.number_input("μ (középérték)", value=1.0, step=0.1)
        sigma = st.number_input("σ (szórás)", value=0.15, min_value=0.0, step=0.01)
        bimodal_shift = st.number_input("Bimodális ±Ω", value=0.8, step=0.1)

        t_max = st.slider("⏱️ Szimuláció hossza (t_max)", 2, 60, 12, step=1)
        dt = st.slider("Δt", 0.01, 0.2, 0.05, step=0.01)

    # ---- Hálózat + Kc
    G = generate_graph(n_nodes, topo, p, k_ws=ws_k)
    A = nx.to_numpy_array(G)
    lam_max = largest_eigenvalue(A)

    # g(0) becslés a választott ω-eloszláshoz (normálhoz pontos)
    sigma_eff = sigma if omega_dist != "Egyenletes" else (np.sqrt(3)*sigma)  # hogy a skála nagyjából hasonló legyen
    g0 = g0_from_sigma(sigma_eff)
    Kc = np.inf if np.isinf(g0) or lam_max <= 0 else 2.0/(np.pi * g0 * lam_max)

    # ---- Preset
    preset = st.segmented_control("Preset", ["Desync", "Near-critical", "Lock-in"], index=1)
    if preset == "Desync":
        K = 0.5 * (Kc if np.isfinite(Kc) else 2.0)
    elif preset == "Near-critical":
        K = 1.0 * (Kc if np.isfinite(Kc) else 2.0)
    else:
        K = 1.6 * (Kc if np.isfinite(Kc) else 2.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("N", f"{n_nodes}")
    col2.metric("λₘₐₓ(G)", f"{lam_max:.2f}")
    col3.metric("Kritikus Kc", "∞" if not np.isfinite(Kc) else f"{Kc:.2f}", delta=None)
    st.info(f"Beállított K = **{K:.2f}**  •  K/Kc = **{(K/Kc):.2f}**" if np.isfinite(Kc) else f"Beállított K = **{K:.2f}**")

    # ---- ω mintavétel
    omega = sample_omega(n_nodes, omega_dist, loc=mu, scale=sigma, bimodal_shift=bimodal_shift)

    # ---- Szimuláció
    if st.button("▶️ Szimuláció indítása", type="primary"):
        t, R, theta_final, theta_frames = simulate_kuramoto(G, K, omega, t_max=float(t_max), dt=float(dt), record=60)

        st.subheader("📈 Szinkronizációs dinamika")
        plot_sync(t, R)

        st.subheader("🌐 3D gráf vizualizáció")
        # időcsúszka a pillanatképekhez
        idx = st.slider("Pillanatkép index", 0, len(theta_frames)-1, len(theta_frames)-1)
        plot_graph_3d(G, theta_frames[idx], title=f"Kuramoto – pillanatkép t ≈ {t[int(idx*len(t)/len(theta_frames))]:.2f}")

        # Export
        st.subheader("💾 Szinkronizációs adatok letöltése")
        df_export = pd.DataFrame({"t": t, "R(t)": R})
        st.download_button("⬇️ CSV letöltése", df_export.to_csv(index=False).encode("utf-8"),
                           file_name="generative_kuramoto.csv", mime="text/csv")

        # Tudományos háttér
        st.markdown("### 📘 Tudományos háttér")
        st.latex(r"\dot{\theta}_i=\omega_i+\frac{K}{N}\sum_{j=1}^N A_{ij}\sin(\theta_j-\theta_i)")
        st.latex(r"R(t)=\left|\frac{1}{N}\sum_{j=1}^N e^{i\theta_j(t)}\right|")
        st.markdown(
            "- \(K_c \approx 2 / [\pi\,g(0)\,\lambda_{\max}(A)]\).  "
            "Ha \(K>K_c\) → **fáziszár**, ha \(K<K_c\) → **dekoherencia**."
        )

        st.subheader("📝 Megfigyelések")
        st.text_area("Mit látsz? Írd ide a tapasztalatokat…", placeholder="Pl.: K≈Kc körül hirtelen nő R(t)…")

# ReflectAI kompatibilitás
app = run
