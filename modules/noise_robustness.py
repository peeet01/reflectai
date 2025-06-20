import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data(show_spinner=False)
def simulate_kuramoto(N, K, T, noise_levels, dt=0.05):
    results = {}
    omega = np.random.normal(0, 1, N)
    theta0 = np.random.uniform(0, 2 * np.pi, N)

    for noise in noise_levels:
        theta = theta0.copy()
        r_vals = []

        for _ in range(T):
            delta_theta = theta[:, None] - theta
            coupling = np.sum(np.sin(delta_theta), axis=1)
            theta += (omega + (K / N) * coupling) * dt
            theta += noise * np.random.normal(0, 1, N) * dt
            r = np.abs(np.sum(np.exp(1j * theta)) / N)
            r_vals.append(r)

        results[np.round(noise, 2)] = np.array(r_vals)

    return results

def plot_results(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for noise, r_vals in results.items():
        ax.plot(r_vals, label=f"Zaj = {noise}", linewidth=2)

    ax.set_title("🎧 Zajtűrés hatása a szinkronizációra")
    ax.set_xlabel("⏱️ Időlépések")
    ax.set_ylabel("🔗 Szinkronizációs index (r)")
    ax.legend()
    ax.grid(True)
    return fig

def run():
    st.subheader("🎛️ Pro szintű zajtűrés vizualizáció – Kuramoto modell")

    N = st.slider("🧠 Oszcillátorok száma", 10, 100, 30)
    K = st.slider("🔗 Kapcsolási erősség", 0.0, 10.0, 2.0)
    T = st.slider("🕒 Szimulációs időlépések", 100, 2000, 500)
    dt = st.slider("📏 Időlépés mérete (dt)", 0.01, 0.1, 0.05)

    noise_levels = st.multiselect(
        "🔉 Zajszintek összehasonlítása (max 4)",
        [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        default=[0.0, 0.3, 0.7]
    )

    if len(noise_levels) == 0 or len(noise_levels) > 4:
        st.warning("⚠️ Kérlek válassz 1–4 zajszintet az összehasonlításhoz.")
        return

    if st.button("▶️ Szimuláció indítása"):
        with st.spinner("Szimuláció fut..."):
            results = simulate_kuramoto(N, K, T, noise_levels, dt)
            fig = plot_results(results)
            st.pyplot(fig)

            # Záró statisztika
            final_sync = {k: round(v[-1], 3) for k, v in results.items()}
            sorted_sync = sorted(final_sync.items())
            summary = "\n".join([f"Zaj = {k}: r = {v}" for k, v in sorted_sync])
            st.markdown("### 🔍 Végső szinkronizációs indexek:")
            st.code(summary)
