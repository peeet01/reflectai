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

        results[noise] = np.array(r_vals)

    return results

def run():
    st.subheader("🎧 Zajtűrés és szinkronizáció vizsgálata (Pro Optimalizált)")

    N = st.slider("🧠 Oszcillátorok száma", 10, 100, 30)
    K = st.slider("🔗 Kapcsolási erősség", 0.0, 10.0, 2.0)
    T = st.slider("⏳ Időlépések száma", 100, 1000, 300)

    noise_levels = st.multiselect("🔉 Választható zajszintek (max 3)", [0.0, 0.1, 0.3, 0.5, 0.7, 1.0], default=[0.0, 0.3, 0.7])
    if len(noise_levels) > 3:
        st.warning("⚠️ Maximum 3 zajszint választható a gyorsabb működéshez.")
        return

    if st.button("▶️ Szimuláció indítása"):
        results = simulate_kuramoto(N, K, T, noise_levels)

        fig, ax = plt.subplots()
        for noise, r in results.items():
            ax.plot(r, label=f"zaj = {noise}", linewidth=2)

        ax.set_title("Szinkronizáció különböző zajszintek mellett")
        ax.set_xlabel("Időlépés")
        ax.set_ylabel("Szinkronizációs index (r)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Válassz paramétereket és indítsd el a szimulációt.")
