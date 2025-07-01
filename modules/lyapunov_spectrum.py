import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

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

def logistic_map(r):
    return lambda x: r * x * (1 - x)

def run():
    st.title("📈 Lyapunov spektrum vizualizáció – Dinamikus rendszerek káosza")

    st.markdown("""
    A Lyapunov-spektrum bemutatja, hogy egy logisztikus leképezés mikor viselkedik determinisztikusan vagy káotikusan a paramétertartományban.
    """)

    r_min = st.slider("🔽 r minimum érték", 2.5, 3.5, 2.5)
    r_max = st.slider("🔼 r maximum érték", 3.5, 4.0, 4.0)
    n_r = st.slider("📊 r értékek száma", 100, 500, 300)
    x0 = st.slider("⚙️ Kezdeti érték (x₀)", 0.0, 1.0, 0.5)
    n_x0 = st.slider("🧮 x₀ minták száma (3D-hez)", 10, 100, 40)

    r_values = np.linspace(r_min, r_max, n_r)
    lyapunov_values = []

    for r in r_values:
        f = logistic_map(r)
        lyap = compute_lyapunov(f, x0)
        lyapunov_values.append(lyap)

    lyapunov_values = np.array(lyapunov_values)

    tabs = st.tabs(["📉 2D Spektrum", "🌐 3D Spektrum"])
    with tabs[0]:
        fig, ax = plt.subplots()
        colors = np.where(lyapunov_values < 0, 'green', 'red')
        ax.scatter(r_values, lyapunov_values, c=colors, s=2)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel("r paraméter")
        ax.set_ylabel("Lyapunov-exponens λ")
        ax.set_title("Lyapunov spektrum – logisztikus térkép")
        st.pyplot(fig)

        df = pd.DataFrame({"r": r_values, "lambda": lyapunov_values})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Eredmények letöltése CSV formátumban", data=csv, file_name="lyapunov_spectrum.csv")

    with tabs[1]:
        r_vals = np.linspace(r_min, r_max, n_r)
        x0_vals = np.linspace(0.01, 0.99, n_x0)
        R, X0 = np.meshgrid(r_vals, x0_vals)
        Z = np.zeros_like(R)

        for i in range(X0.shape[0]):
            for j in range(X0.shape[1]):
                f = logistic_map(R[i, j])
                Z[i, j] = compute_lyapunov(f, X0[i, j])

        fig3d = go.Figure(data=[go.Surface(
            z=Z,
            x=R,
            y=X0,
            colorscale="Inferno",
            colorbar=dict(title="λ", thickness=15),
            lighting=dict(ambient=0.5, diffuse=1.0, roughness=0.5),
            opacity=0.95
        )])

        fig3d.update_layout(
            title="🌐 3D Lyapunov-spektrum (r és x₀ szerint)",
            scene=dict(
                xaxis_title="r",
                yaxis_title="x₀",
                zaxis_title="λ",
                camera=dict(eye=dict(x=1.3, y=1.3, z=0.8))
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_dark"
        )

        st.plotly_chart(fig3d, use_container_width=True)

    with st.expander("📘 Tudományos háttér – Mi az a Lyapunov-exponens?"):
        st.markdown(r"""
        A **Lyapunov-exponens** egy kvantitatív mutató, amely leírja, hogy egy dinamikus rendszer **mennyire érzékeny a kezdeti feltételekre**.

        ---
        **Alapgondolat:**  
        Ha két kiindulási érték nagyon közel van egymáshoz, a Lyapunov-exponens megmutatja, mennyire gyorsan távolodnak el egymástól idővel.

        ---
        ### 📐 Matematikai definíció:
        $$ 
        \lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} \ln \left| \frac{df(x_i)}{dx} \right| 
        $$

        - **λ < 0**: stabil viselkedés (periodikus, fixpont)
        - **λ = 0**: neutrális viselkedés
        - **λ > 0**: káosz, erősen érzékeny rendszer

        ---
        A logisztikus térképen ez a káosz határait és jelenlétét segít feltérképezni, gyakori eszköze a nemlineáris dinamika vizsgálatának.
        """)

# Kötelező ReflectAI-kompatibilitás
app = run
