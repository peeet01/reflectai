import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data(show_spinner=False)
def run_kuramoto_sim(N, K, T, noise, dt=0.05):
    omega = np.random.normal(0, 1, N)
    theta = np.random.uniform(0, 2 * np.pi, N)
    r_vals = []

    for _ in range(T):
        delta_theta = theta[:, None] - theta
        coupling = np.sum(np.sin(delta_theta), axis=1)
        theta += (omega + (K / N) * coupling) * dt
        theta += noise * np.random.normal(0, 1, N) * dt
        r = np.abs(np.sum(np.exp(1j * theta)) / N)
        r_vals.append(r)

    return np.array(r_vals)

def run():
    st.subheader("🎧 Zajhatás vizsgálata – Kuramoto modell (Optimalizált)")

    N = st.slider("🧠 Oszcillátorok száma", 10, 200, 50)
    K = st.slider("🔗 Kapcsolási erősség", 0.0, 10.0, 2.0)
    T = st.slider("⏳ Időlépések száma", 100, 1500, 300)
    noise = st.slider("🔉 Zajszint", 0.0, 1.0, 0.2, step=0.05)

    if st.button("▶️ Szimuláció indítása"):
        r = run_kuramoto_sim(N, K, T, noise)

        fig, ax = plt.subplots()
        ax.plot(r, color='purple', linewidth=2)
        ax.set_title(f"Szinkronizáció zaj mellett (zaj = {noise})")
        ax.set_xlabel("Időlépés")
        ax.set_ylabel("Szinkronizációs index (r)")
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Kattints a szimuláció indítására.")
