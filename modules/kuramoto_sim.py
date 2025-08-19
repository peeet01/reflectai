import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pandas as pd

# --- Kuramoto lépés ---
def kuramoto_step(theta, K, A, omega, dt):
    N = len(theta)
    dtheta = omega + (K / N) * np.sum(A * np.sin(theta[:, None] - theta), axis=1)
    return theta + dt * dtheta

# --- Szinkronizációs index ---
def compute_order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))

# --- Hálózat generálás ---
def generate_graph(N, graph_type, p=0.3, k=4, m=2):
    if graph_type == "Teljes":
        return nx.complete_graph(N)
    elif graph_type == "Véletlen (Erdős–Rényi)":
        return nx.erdos_renyi_graph(N, p=p, seed=42)
    elif graph_type == "Kis világ (Watts–Strogatz)":
        k = max(2, min(k, N-1))
        if k % 2 == 1:  # WS: a gyűrűn általában páros k a kerek szomszédsághoz
            k += 1
        return nx.watts_strogatz_graph(N, k=k, p=p, seed=42)
    elif graph_type == "Skálafüggetlen (Barabási–Albert)":
        m = max(1, min(m, N-1))
        return nx.barabasi_albert_graph(N, m=m, seed=42)
    else:
        return nx.complete_graph(N)

# --- 3D gráf kirajzolás ---
def plot_graph_3d(G, theta, palette, layout_type="Spring (hálózati kutatás – alapértelmezett)"):
    # Elrendezés kiválasztása
    if layout_type.startswith("Spring"):
        pos = nx.spring_layout(G, dim=3, seed=42)  # 3D erő-alapú, hálózatkutatásban bevett
        node_xyz = np.array([pos[n] for n in G.nodes()])
    else:
        # Kör elrendezés (klasszikus tankönyvi ábra): 2D -> 3D (z=0)
        pos2d = nx.circular_layout(G)
        node_xy = np.array([pos2d[n] for n in G.nodes()])
        zeros = np.zeros((node_xy.shape[0], 1))
        node_xyz = np.hstack([node_xy, zeros])

    # Élek koordinátái
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = node_xyz[u]
        x1, y1, z1 = node_xyz[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    # Fázist 0..2π közé tekerjük vizualizációhoz
    theta_wrapped = (theta % (2*np.pi))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="lightgray", width=1),
        opacity=0.35,
        name="Kötések"
    ))
    fig.add_trace(go.Scatter3d(
        x=node_xyz[:, 0], y=node_xyz[:, 1], z=node_xyz[:, 2],
        mode="markers",
        marker=dict(
            size=7,
            color=theta_wrapped,
            colorscale=palette,
            opacity=0.9,
            line=dict(color="black", width=0.5)
        ),
        name="Oszcillátorok"
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
        margin=dict(l=0, r=0, b=0, t=32),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# --- App ---
def run():
    st.set_page_config(layout="wide")
    st.title("🧠 Kuramoto Szinkronizáció – Dinamikus Oszcillátor Modell")

    st.markdown(
        "A **Kuramoto-modell** több oszcillátor fázisának szinkronizációját írja le hálózatokon."
    )

    # Oldalsáv – paraméterek
    st.sidebar.header("⚙️ Szimulációs paraméterek")
    N = st.sidebar.slider("Oszcillátorok száma (N)", 5, 200, 50, step=1)
    graph_type = st.sidebar.selectbox(
        "Hálózat típusa",
        ["Teljes", "Véletlen (Erdős–Rényi)", "Kis világ (Watts–Strogatz)", "Skálafüggetlen (Barabási–Albert)"]
    )

    # Topológia-specifikus beállítások
    p = 0.3; k = 4; m = 2
    if graph_type in ["Véletlen (Erdős–Rényi)", "Kis világ (Watts–Strogatz)"]:
        p = st.sidebar.slider("Rewire/él valószínűsége (p)", 0.0, 1.0, 0.3, 0.01)
    if graph_type == "Kis világ (Watts–Strogatz)":
        k = st.sidebar.slider("Gyűrű-szomszédok száma (k)", 2, min(20, N-1), 4, step=1)
    if graph_type == "Skálafüggetlen (Barabási–Albert)":
        m = st.sidebar.slider("Új csomópont élszáma (m)", 1, min(10, N-1), 2, step=1)

    K = st.sidebar.slider("Kapcsolódási erősség (K)", 0.0, 10.0, 2.0, 0.1)
    steps = st.sidebar.slider("Iterációk száma", 100, 5000, 1000, 100)
    dt = st.sidebar.slider("Időlépés (dt)", 0.005, 0.2, 0.05, 0.005)
    palette = st.sidebar.selectbox("Színséma (3D)", ["Turbo", "Viridis", "Electric", "Hot", "Rainbow"])

    # 👇 ÚJ: Elrendezés-választó + magyarázat kapcsoló
    layout_type = st.sidebar.selectbox(
        "Vizualizációs elrendezés",
        ["Spring (hálózati kutatás – alapértelmezett)", "Kör (klasszikus tankönyvi)"]
    )
    show_disclaimer = st.sidebar.checkbox("Magyarázó szöveg megjelenítése a grafikon felett", value=True)

    # Szimuláció
    np.random.seed(42)
    theta = np.random.uniform(0, 2*np.pi, N)
    # frekvenciák: zero-mean, skálázható szórás
    omega_sigma = st.sidebar.slider("Sajátfrekvencia szórás (σ_ω)", 0.0, 2.0, 1.0, 0.1)
    omega = np.random.normal(loc=0.0, scale=omega_sigma, size=N)

    G = generate_graph(N, graph_type, p=p, k=k, m=m)
    A = nx.to_numpy_array(G)

    order_params = []
    for _ in range(steps):
        theta = kuramoto_step(theta, K, A, omega, dt)
        order_params.append(compute_order_parameter(theta))

    # Kritikus K becslés (all-to-all, normális ω feltételezés)
    # g(0) = 1/(sqrt(2π) * σ), Kc ≈ 2/(π*g(0)) = 2*sqrt(2π)*σ/π
    if omega_sigma > 0:
        Kc_est = 2*np.sqrt(2*np.pi)*omega_sigma/np.pi
        tip = "felette" if K >= Kc_est else "alatta"
        st.info(f"Elméleti K_c (all-to-all közelítés): ~ **{Kc_est:.2f}** – a választott K **{tip}** van.")
    else:
        st.info("ω szórása 0 – ideális esetben már kis K mellett is szinkronizáció várható.")

    # 2D R(t)
    st.subheader("📈 Szinkronizációs index R(t)")
    st.line_chart(order_params)

    # 💬 Magyarázó szöveg az elrendezéshez (disclaimer)
    if show_disclaimer:
        if layout_type.startswith("Spring"):
            st.markdown(
                "> *A csomópontok 3D **erő-alapú elrendezésben** láthatók. Hálózatkutatásban ezt használjuk, "
                "mert a **gráf szerkezete** a lényeges, nem a geometriai körpozíció.*"
            )
        else:
            st.markdown(
                "> *Klasszikus **kör elrendezés**: didaktikus, tankönyvi nézet. A dinamika ugyanaz, "
                "csak a megjelenítés változik.*"
            )

    # 3D hálózat
    st.subheader("🌐 3D vizualizáció – Oszcillátor fázisok színkóddal")
    plot_graph_3d(G, theta, palette, layout_type=layout_type)

    # Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")
    st.latex(r"\frac{d\theta_i}{dt}=\omega_i+\frac{K}{N}\sum_{j=1}^N A_{ij}\sin(\theta_j-\theta_i)")
    st.markdown(
        "- **$\\theta_i$**: az *i*-edik oszcillátor fázisa  \n"
        "- **$\\omega_i$**: sajátfrekvencia  \n"
        "- **$A_{ij}$**: szomszédsági mátrix  \n"
        "- **$K$**: globális kapcsolódási erősség  \n"
        "Az **order parameter** ($R$):"
    )
    st.latex(r"R(t)=\left|\frac{1}{N}\sum_{j=1}^N e^{i\theta_j(t)}\right|")

    # Export
    df_export = pd.DataFrame({"Időlépés": np.arange(1, steps+1), "Szinkronizáció (R)": order_params})
    st.subheader("💾 Szinkronizációs adatok letöltése")
    st.download_button("⬇️ CSV letöltése", df_export.to_csv(index=False).encode("utf-8"),
                       "kuramoto_sync.csv", "text/csv")

    st.subheader("📝 Megfigyelések és jegyzetek")
    st.text_area("Mit tapasztaltál a szinkronizáció során?", placeholder="Írd ide...")

# ReflectAI kompat
app = run
