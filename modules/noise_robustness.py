import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def simulate_sync(N, K, T, noise_level, dt=0.05):
    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2*np.pi, N)
    r_values = []

    for _ in range(T):
        dtheta = theta[:, None] - theta
        interaction = np.sum(np.sin(dtheta), axis=1)
        theta += (omega + (K / N) * interaction) * dt
        theta += noise_level * np.random.normal(0, 1, N) * dt
        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        r_values.append(r)

    return np.array(r_values)

def run():
    st.subheader("🧪 Zajtűrés szimuláció (Gyorsított Pro változat)")

    N = st.slider("🧠 Oszcillátorok száma", 5, 50, 20)
    K = st.slider("🔗 Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    T = st.slider("📈 Időlépések", 50, 500, 200)
    dt = st.slider("📏 Időlépés mérete", 0.01, 0.1, 0.05)

    noise_levels = [0.0, 0.1, 0.5, 1.0]
    r_matrix = []

    with st.spinner("Szimuláció..."):
        for noise in noise_levels:
            r_vals = simulate_sync(N, K, T, noise, dt)
            r_matrix.append(r_vals)

    # Ábra
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, noise in enumerate(noise_levels):
        ax.plot(r_matrix[i], label=f"Zaj={noise}", linewidth=2)
    ax.set_title("📊 Szinkronizáció időbeli alakulása zajszintek szerint")
    ax.set_xlabel("Időlépés")
    ax.set_ylabel("r-index")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Stat
    st.markdown("### 📋 Átlagos szinkronizációs értékek")
    for i, noise in enumerate(noise_levels):
        avg_r = np.round(np.mean(r_matrix[i][-50:]), 3)
        st.write(f"🔉 Zaj={noise} → Átlagos r-index (utolsó 50 lépés): `{avg_r}`")
