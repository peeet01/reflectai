import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from modules.data_upload import get_uploaded_data, show_data_overview


def kuramoto_fast(N, T, dt, K, noise_std):
    theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.random.normal(0, 1, N)
    order_params = []

    for _ in range(T):
        mean_field = np.mean(np.exp(1j * theta))
        coupling = np.imag(mean_field * np.exp(-1j * theta))
        noise = np.random.normal(0, noise_std, N)
        theta += (omega + K * coupling + noise) * dt
        r = np.abs(mean_field)
        order_params.append(r)

    return np.mean(order_params)


def run():
    st.title("🔊 Zajtűrés és szinkronizációs robusztusság")

    st.markdown("""
    Ez a modul azt vizsgálja, hogy különböző zaj- és kapcsolati erősségek mellett mennyire marad szinkronban egy oszcillátorhálózat.
    Használhatsz saját adatot is (pl. fázisokat tartalmazó `.csv` fájlt), vagy alapértelmezett szimulációt.
    """)

    df = get_uploaded_data(required_columns=None, allow_default=False)

    if df is not None:
        st.success("✅ Saját adat betöltve.")
        show_data_overview(df)

        st.info("🔧 Egyedi adatfeltöltés jelenleg még nem implementált — a jelenlegi verzió csak szimulációval működik.")
        st.stop()

    # Ha nincs adat, szimulációs móddal folytatjuk
    st.markdown("### ⚙️ Szimuláció paraméterek")

    N = st.slider("🧠 Oszcillátorok száma", 5, 50, 20)
    T = st.slider("⏱️ Iterációk száma", 50, 300, 150)
    dt = st.slider("🕒 Időlépés", 0.01, 0.1, 0.03)
    num_K = st.slider("📈 K felbontás", 5, 15, 8)
    num_noise = st.slider("📉 Zaj felbontás", 5, 15, 8)

    K_vals = np.linspace(0.0, 10.0, num_K)
    noise_vals = np.linspace(0.0, 2.0, num_noise)
    R_matrix = np.zeros((num_K, num_noise))

    progress = st.progress(0.0, text="Szimuláció fut...")

    for i, K in enumerate(K_vals):
        for j, noise in enumerate(noise_vals):
            R_matrix[i, j] = kuramoto_fast(N, T, dt, K, noise)
        progress.progress((i + 1) / num_K, text=f"{int((i + 1) / num_K * 100)}% kész")

    st.markdown("### 🔍 Szinkronizációs index hőtérkép")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(R_matrix, origin='lower', aspect='auto',
                   extent=[noise_vals[0], noise_vals[-1], K_vals[0], K_vals[-1]],
                   cmap='plasma')
    ax.set_xlabel("Zaj szórása (σ)")
    ax.set_ylabel("Kapcsolási erősség (K)")
    ax.set_title("🌀 Átlagos szinkronizációs index (r)")
    plt.colorbar(im, ax=ax, label="Átlagos r")
    st.pyplot(fig)
