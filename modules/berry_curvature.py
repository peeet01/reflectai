import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def compute_berry_curvature(kx, ky, delta=0.1):
    """
    Egyszerű 2D Dirac-modell Berry-görbületének kiszámítása.
    """
    d = np.array([
        np.sin(kx),
        np.sin(ky),
        delta + np.cos(kx) + np.cos(ky)
    ])
    norm = np.linalg.norm(d)
    d_hat = d / (norm + 1e-8)
    curvature = 0.5 * d_hat[2] / (norm**2 + 1e-8)
    return curvature

def run():
    st.title("🌀 Berry-görbület és topológiai fázis")
    st.markdown("""
    Ez a modul a **Berry-görbületet** vizualizálja a 2D Brillouin-zónában,  
    valamint kiszámolja az **integrált Berry-fázist** és a közelítő **Chern-számot**.
    """)

    st.sidebar.header("🧮 Paraméterek")
    N = st.sidebar.slider("Rácspontok tengelyenként", 20, 150, 80, step=10)
    delta = st.sidebar.slider("Delta érték (tömeg tag)", 0.0, 1.0, 0.1, step=0.01)

    kx_vals = np.linspace(-np.pi, np.pi, N)
    ky_vals = np.linspace(-np.pi, np.pi, N)
    dk = kx_vals[1] - kx_vals[0]

    curvature = np.zeros((N, N))
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            curvature[j, i] = compute_berry_curvature(kx, ky, delta=delta)

    # 🔷 Ábra 1: Berry-görbület kontúrtérkép
    st.subheader("📊 Berry-görbület a Brillouin-zónában")
    fig1, ax1 = plt.subplots()
    c = ax1.contourf(kx_vals, ky_vals, curvature, levels=50, cmap='coolwarm')
    fig1.colorbar(c, ax=ax1, label="Berry-görbület")
    ax1.set_xlabel("kx")
    ax1.set_ylabel("ky")
    ax1.set_title("Berry-görbület kontúrtérkép")
    st.pyplot(fig1)

    # 🔸 Berry-fázis integrál (Chern-szám)
    berry_phase = np.sum(curvature) * dk * dk
    chern_number = berry_phase / (2 * np.pi)

    # 🔷 Ábra 2: Fázisintegrál értéke
    st.subheader("📈 Integrált Berry-fázis és Chern-szám")
    fig2, ax2 = plt.subplots()
    ax2.bar(["Berry-fázis ∫F", "Chern-szám (∫F / 2π)"], [berry_phase, chern_number], color=["purple", "orange"])
    ax2.set_ylabel("Érték")
    ax2.set_title("Topológiai invariánsok")
    st.pyplot(fig2)

    # 🔎 Értelmezés
    st.markdown("---")
    st.markdown(f"🔺 **Delta érték**: `{delta}`")
    st.markdown(f"📐 **Integrált Berry-fázis**: `{berry_phase:.4f}`")
    st.markdown(f"🔢 **Közelítő Chern-szám**: `{chern_number:.4f}`")

    if abs(chern_number) > 0.4:
        st.success("🎯 Nemtriviális topológiai fázis!")
    else:
        st.info("🧱 Triviális topológia (nincs élállapot)")

# Kötelező ReflectAI belépési pont
app = run
