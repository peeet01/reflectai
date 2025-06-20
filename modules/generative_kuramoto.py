import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def run(epochs=1000, noise_dim=10, num_oscillators=10):
    st.header("🎲 Generatív Kuramoto szinkronizáció")

    # Egyszerű referencia fázisminták Kuramoto-modell alapján
    def generate_real_phases(n):
        base = np.linspace(0, 2 * np.pi, n)
        noise = np.random.normal(0, 0.3, n)
        return (base + noise) % (2 * np.pi)

    # Generator: véletlen zaj -> fázisok (egyszerű FF hálózat szimuláció)
    def generator(z):
        return (np.sin(z @ np.random.randn(z.shape[1], num_oscillators)) + 1) * np.pi

    real_samples = np.array([generate_real_phases(num_oscillators) for _ in range(100)])
    fake_samples = np.array([generator(np.random.randn(1, noise_dim))[0] for _ in range(100)])

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={'projection': 'polar'})
    axs[0].set_title("🎯 Valódi Kuramoto fázisok")
    axs[1].set_title("🧪 Generált fázisok")

    for sample in real_samples[:10]:
        axs[0].plot(sample, np.ones_like(sample), 'o')
    for sample in fake_samples[:10]:
        axs[1].plot(sample, np.ones_like(sample), 'o')

    st.pyplot(fig)
    st.markdown("🔄 Ez a modell generált fázismintázatokat állít elő, és összehasonlítja azokat a valós Kuramoto kimenetekkel.")
