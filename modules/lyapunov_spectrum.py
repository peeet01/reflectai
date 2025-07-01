import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

# Lyapunov-exponens számítása
def compute_lyapunov(f, x0, delta=1e-8, steps=1000):
    x = x0
    d = delta
    lyapunov_sum = 0.0
    for _ in range(steps):
        x1 = f(x)
        x2 = f(x + d)
        d = np.abs(x2 - x1)
        d = d if d != 0 else 1e-8
        lyapunov_sum += np.log(np.abs(d / delta))
        x = x1
    return lyapunov_sum / steps

# Logisztikus leképezés
def logistic_map(r):
    return lambda x: r * x * (1 - x)

def run():
    st.title("🌐 Lyapunov Spektrum – 2D és 3D vizualizáció")

    st.markdown("A logisztikus leképezés stabilitásának és káoszának vizsgálata Lyapunov-exponens alapján.")

    # Paraméterek
    r_min = st.slider("🔽 r minimum", 2.5, 3.5, 2.5)
    r_max = st.slider("🔼 r maximum", 3.5, 4.0, 4.0)
    n_points = st.slider("📊 r pontok száma", 100, 1000, 500)
    x0_min = st.slider("⚙️ x₀ minimum", 0.0, 1.0, 0.1)
    x0_max = st.slider("⚙️ x₀ maximum", 0.0, 1.0, 0.9)
    x0_points = st.slider("📈 x₀ pontok száma", 10, 100, 40)

    r_values = np.linspace(r_min, r_max, n_points)
    x0_values = np.linspace(x0_min, x0_max, x0_points)
    spectrum = np.zeros((x0_points, n_points))

    # Számítás
    with st.spinner("⏳ Spektrum számítása..."):
        for i, x0 in enumerate(x0_values):
            for j, r in enumerate(r_values):
                f = logistic_map(r)
                spectrum[i, j] = compute_lyapunov(f, x0)

    # Választható nézet
    view = st.radio("🖼️ Nézet", ["2D Spektrum", "3D Spektrum"])

    if view == "2D Spektrum":
        fig, ax = plt.subplots()
        avg_lyap = np.mean(spectrum, axis=0)
        colors = np.where(avg_lyap < 0, 'green', 'red')
        ax.scatter(r_values, avg_lyap, c=colors, s=2)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel("r paraméter")
        ax.set_ylabel("Átlagos λ")
        ax.set_title("Lyapunov spektrum – logisztikus térkép (2D)")
        st.pyplot(fig)

    else:
        r_mesh, x0_mesh = np.meshgrid(r_values, x0_values)
        fig3d = go.Figure(data=[go.Surface(
            z=spectrum,
            x=r_mesh,
            y=x0_mesh,
            colorscale='Inferno',
            colorbar=dict(title="λ"),
            lighting=dict(ambient=0.5, diffuse=0.9, roughness=0.2),
        )])

        fig3d.update_layout(
            title="Lyapunov spektrum – logisztikus leképezés (3D)",
            scene=dict(
                xaxis_title="r",
                yaxis_title="x₀",
                zaxis_title="λ (Lyapunov)"
            ),
            margin=dict(l=10, r=10, b=10, t=40)
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # CSV export
    df = pd.DataFrame(spectrum, index=[f"{x0:.3f}" for x0 in x0_values], columns=[f"{r:.3f}" for r in r_values])
    csv = df.to_csv().encode("utf-8")
    st.download_button("⬇️ Eredmények letöltése CSV-ben", data=csv, file_name="lyapunov_3d_spectrum.csv")

    # Tudományos háttér
    with st.expander("📘 Tudományos háttér – Mi az a Lyapunov-exponens?"):
        st.markdown("""
        A **Lyapunov-exponens** egy dinamikus rendszer **szenzitivitását** méri a kezdeti értékekre.
        
        ---
        **Alapgondolat:**  
        Ha két nagyon hasonló kezdeti állapot gyorsan eltávolodik egymástól, a rendszer kaotikus.

        ---
        ### 📐 Matematikai formula:
        $$
        \\lambda = \\lim_{n \\to \\infty} \\frac{1}{n} \\sum_{i=1}^{n} \\ln \\left| \\frac{df(x_i)}{dx} \\right|
        $$

        ---
        **Értelmezés:**
        - **λ < 0**: stabil, determinisztikus viselkedés
        - **λ = 0**: neutrális állapot
        - **λ > 0**: kaotikus dinamika

        A 3D spektrum lehetővé teszi a kezdeti állapot és rendszerparaméter együttes hatásának vizsgálatát.
        """)

# Kötelező ReflectAI-kompatibilitás
app = run
