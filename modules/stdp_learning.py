import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def stdp_function(delta_t, A_plus, A_minus, tau_plus, tau_minus):
    return np.where(
        delta_t > 0,
        A_plus * np.exp(-delta_t / tau_plus),
        -A_minus * np.exp(delta_t / tau_minus)
    )


def simulate_spike_pairs(n_pairs, t_range_ms, jitter=5):
    np.random.seed(0)
    pre_spikes = np.random.uniform(-t_range_ms, t_range_ms, size=n_pairs)
    post_spikes = pre_spikes + np.random.normal(loc=0, scale=jitter, size=n_pairs)
    delta_t = post_spikes - pre_spikes
    return delta_t


def run():
    st.set_page_config(layout="wide")
    st.title("⏱️ STDP – Spike-Timing Dependent Plasticity")

    st.markdown("""
    A **Spike-Timing Dependent Plasticity (STDP)** egy biológiailag ihletett tanulási szabály,  
    amely a szinaptikus erőket a tüskék időzítésének függvényében módosítja.

    Ez az alkalmazás lehetővé teszi az STDP viselkedésének szimulációját, vizualizációját és exportálását.
    """)

    # 🎚️ Paraméterek
    st.sidebar.header("🛠️ Paraméterek")
    A_plus = st.sidebar.slider("Erősítés (A⁺)", 0.01, 1.0, 0.1)
    A_minus = st.sidebar.slider("Gyengítés (A⁻)", 0.01, 1.0, 0.12)
    tau_plus = st.sidebar.slider("τ⁺ (időkonstans, ms)", 1.0, 50.0, 20.0)
    tau_minus = st.sidebar.slider("τ⁻ (időkonstans, ms)", 1.0, 50.0, 20.0)
    t_range = st.sidebar.slider("Δt tartomány (ms)", 10, 200, 100)
    n_spikes = st.sidebar.slider("Szintetikus spike-párok száma", 10, 500, 100)
    jitter = st.sidebar.slider("Jitter a post-synaptic tüskére (ms)", 0.0, 20.0, 5.0)

    # 📉 STDP görbe
    delta_t = np.linspace(-t_range, t_range, 500)
    delta_w = stdp_function(delta_t, A_plus, A_minus, tau_plus, tau_minus)

    st.subheader("📉 STDP görbe")
    fig, ax = plt.subplots()
    ax.plot(delta_t, delta_w, color='orange')
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel("Időeltolódás (Δt, ms)")
    ax.set_ylabel("Súlyváltozás (Δw)")
    ax.set_title("Szinaptikus súlyváltozás függése Δt-től")
    st.pyplot(fig)

    # 🌐 3D vizualizáció
    st.subheader("🌐 3D vizualizáció – STDP tér")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=delta_t,
        y=delta_w,
        z=np.zeros_like(delta_t),
        mode='lines',
        line=dict(color=delta_w, colorscale='Viridis', width=6)
    )])
    fig3d.update_layout(
        scene=dict(xaxis_title="Δt (ms)", yaxis_title="Δw", zaxis_title="z=0"),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500
    )
    st.plotly_chart(fig3d)

    # 🔬 Spike-pár szimuláció
    st.subheader("🔬 Szintetikus spike-párok szimulációja")
    spike_dts = simulate_spike_pairs(n_spikes, t_range, jitter)
    spike_dws = stdp_function(spike_dts, A_plus, A_minus, tau_plus, tau_minus)

    fig2, ax2 = plt.subplots()
    ax2.hist(spike_dts, bins=30, alpha=0.6, label="Δt eloszlás")
    ax2.set_xlabel("Δt (ms)")
    ax2.set_ylabel("Gyakoriság")
    ax2.set_title("Tüskepárok időeltolódása")
    st.pyplot(fig2)

    st.markdown(f"**Átlagos súlyváltozás:** {np.mean(spike_dws):.4f}")

    # 📁 CSV export
    st.subheader("📥 Adatok letöltése")
    df_export = pd.DataFrame({
        "delta_t": delta_t,
        "delta_w": delta_w
    })
    df_spikes = pd.DataFrame({
        "spike_pair_delta_t": spike_dts,
        "spike_pair_delta_w": spike_dws
    })
    csv_main = df_export.to_csv(index=False).encode("utf-8")
    csv_spikes = df_spikes.to_csv(index=False).encode("utf-8")

    st.download_button("⬇️ STDP görbe letöltése", data=csv_main, file_name="stdp_curve.csv")
    st.download_button("⬇️ Spike-párok exportja", data=csv_spikes, file_name="stdp_spike_pairs.csv")

    # 📘 Tudományos háttér
    st.markdown("### 📚 Tudományos háttér")

    st.latex(r"""
    Δw(Δt) = 
    \begin{cases}
    A^+ \cdot e^{-Δt / τ^+} & \text{ha } Δt > 0 \\
    -A^- \cdot e^{Δt / τ^-} & \text{ha } Δt < 0
    \end{cases}
    """)

    st.markdown("""
    A **Spike-Timing Dependent Plasticity** kulcsszerepet játszik az agyi tanulási folyamatokban.  
    - **Pozitív időeltolódás (Δt > 0)** esetén erősödik a szinapszis (LTP).  
    - **Negatív időeltolódás (Δt < 0)** esetén gyengül (LTD).  
    Az **STDP görbe** alakját az \( A^+, A^- \) és \( τ^+, τ^- \) paraméterek határozzák meg.

    **Alkalmazás:** tanulási szabály nem felügyelt neurális modellekben, biológiailag inspirált tanulás szimulációja.
    """)

# ReflectAI kompatibilitás
app = run
