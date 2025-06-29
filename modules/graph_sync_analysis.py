import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import pandas as pd

def kuramoto_step(theta, omega, A, K, dt):
    N = len(theta)
    dtheta = omega + (K / N) * np.sum(A * np.sin(np.subtract.outer(theta, theta)), axis=1)
    return theta + dtheta * dt

def run_simulation(G, steps, dt, K):
    N = len(G)
    theta = np.random.rand(N) * 2 * np.pi
    omega = np.random.randn(N)
    A = nx.to_numpy_array(G)

    r_values = []
    theta_history = []

    for _ in range(steps):
        theta = kuramoto_step(theta, omega, A, K, dt)
        r = np.abs(np.mean(np.exp(1j * theta)))
        r_values.append(r)
        theta_history.append(theta.copy())
    return r_values, theta_history

def draw_graph(G, theta=None, title="Gráf vizualizáció"):
    pos = nx.spring_layout(G, seed=42)
    if theta is not None:
        norm_theta = (theta % (2 * np.pi)) / (2 * np.pi)
        node_colors = cm.hsv(norm_theta)
    else:
        node_colors = 'lightblue'

    fig, ax = plt.subplots()
    nx.draw(G, pos, node_color=node_colors, edge_color='gray', with_labels=True, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def run():
    st.title("🔗 Gráfalapú szinkronanalízis – Kuramoto-modell")
    st.markdown("Vizsgáld meg, hogyan alakul a szinkronizáció különböző gráfstruktúrákon.")

    with st.sidebar:
        st.header("⚙️ Paraméterek")
        graph_type = st.selectbox("Gráftípus", ["Erdős–Rényi", "Kör", "Rács", "Teljes gráf"])
        N = st.slider("Csomópontok száma", 5, 100, 30)
        K = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
        steps = st.slider("Lépések száma", 10, 1000, 300)
        dt = st.slider("Időlépés (dt)", 0.001, 0.1, 0.01)
        er_p = st.slider("Erdős–Rényi élvalószínűség", 0.05, 1.0, 0.1, step=0.05)

    if st.button("▶️ Szimuláció indítása"):
        if graph_type == "Erdős–Rényi":
            G = nx.erdos_renyi_graph(N, er_p)
        elif graph_type == "Kör":
            G = nx.cycle_graph(N)
        elif graph_type == "Rács":
            side = int(np.sqrt(N))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
        elif graph_type == "Teljes gráf":
            G = nx.complete_graph(N)

        with st.spinner("Szimuláció fut..."):
            start = time.time()
            r_values, theta_hist = run_simulation(G, steps, dt, K)
            end = time.time()

        st.success(f"✅ Szimuláció kész ({end - start:.2f} sec)")
        st.metric("📈 Végső szinkronizáció (r)", f"{r_values[-1]:.3f}")
        st.metric("📊 Átlagos szinkronizáció (r)", f"{np.mean(r_values):.3f}")

        # r(t) grafikon
        st.subheader("📉 Szinkronizáció időbeli alakulása")
        fig1, ax1 = plt.subplots()
        ax1.plot(r_values)
        ax1.set_xlabel("Időlépés")
        ax1.set_ylabel("Szinkronizáció (r)")
        ax1.set_title("Kuramoto – r(t)")
        st.pyplot(fig1)

        # Kezdet és vég vizualizáció
        st.subheader("🧠 Kezdeti gráf vizualizáció")
        draw_graph(G, theta_hist[0], "Kezdeti fázisok")

        st.subheader("🧠 Végállapot gráf vizualizáció")
        draw_graph(G, theta_hist[-1], "Végső fázisok")

        # Animált léptetés
        if st.checkbox("🎞️ Gráf animáció megtekintése", value=False):
            st.subheader("🎬 Szinkronizáció alakulása lépésenként")
            step_to_show = st.slider("Animáció lépése", 0, steps - 1, 0)
            draw_graph(G, theta_hist[step_to_show], f"{step_to_show}. lépés")

        # CSV letöltés
        df = pd.DataFrame({
            "step": list(range(len(r_values))),
            "r_value": r_values
        })
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 r(t) exportálása CSV-ként", csv, file_name="kuramoto_r_values.csv")

        # Jegyzet
        st.subheader("📝 Jegyzetek")
        notes = st.text_area("Írd le megfigyeléseid:", height=150)
        if notes:
            st.download_button("💾 Jegyzet mentése", notes, file_name="kuramoto_notes.txt")

    with st.expander("📚 Elméleti háttér"):
        st.markdown(r"""
        A **Kuramoto-modell** egy nemlineáris differenciálegyenlet-rendszer, amelyet oszcillátorok szinkronizációjának modellezésére használnak.

        Az egyes csomópontok saját frekvenciával rendelkeznek (`ω`), és a kapcsolatok (gráf élei) mentén szinkronizálnak a többiekkel.  
        A rendszer szinkronizációját az `r` érték méri:  
        - `r = 1` → teljes szinkronizáció  
        - `r ~ 0` → kaotikus állapot

        #### Alkalmazási területek:
        - Neurális oszcillációk
        - Elektromos hálózatok
        - Biológiai ritmusok (pl. szívsejtek)

        📖 *Kuramoto, Y. (1975). "Self-entrainment of a population of coupled non-linear oscillators"*
        """)

# Kötelező ReflectAI kompatibilitáshoz
app = run
