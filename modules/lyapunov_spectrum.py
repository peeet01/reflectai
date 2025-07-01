import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_lyapunov(f, x0, delta=1e-8, steps=1000):
    x = x0
    d = delta
    lyapunov_sum = 0.0

    for _ in range(steps):
        x1 = f(x)
        x2 = f(x + d)

        d = np.abs(x2 - x1)
        d = d if d != 0 else 1e-8  # elkerüljük a log(0)-t
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
    n_points = st.slider("📊 Mintavételezési pontok száma", 100, 1000, 500)
    x0 = st.slider("⚙️ Kezdeti érték (x₀)", 0.0, 1.0, 0.5)

    r_values = np.linspace(r_min, r_max, n_points)
    lyapunov_values = []

    for r in r_values:
        f = logistic_map(r)
        lyap = compute_lyapunov(f, x0)
        lyapunov_values.append(lyap)

    lyapunov_values = np.array(lyapunov_values)

    fig, ax = plt.subplots()
    colors = np.where(lyapunov_values < 0, 'green', 'red')
    ax.scatter(r_values, lyapunov_values, c=colors, s=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("r paraméter")
    ax.set_ylabel("Lyapunov-exponens λ")
    ax.set_title("Lyapunov spektrum – logisztikus térkép")
    st.pyplot(fig)

    # CSV export
    df = pd.DataFrame({"r": r_values, "lambda": lyapunov_values})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Eredmények letöltése CSV formátumban", data=csv, file_name="lyapunov_spectrum.csv")

    # Tudományos háttér
    with st.expander("📘 Tudományos háttér – Mi az a Lyapunov-exponens?"):
        st.markdown("""
        A **Lyapunov-exponens** egy kvantitatív mutató, amely leírja, hogy egy dinamikus rendszer **mennyire érzékeny a kezdeti feltételekre**.

        ---
        **Alapgondolat:**  
        Ha két kiindulási érték nagyon közel van egymáshoz, a Lyapunov-exponens megmutatja, mennyire gyorsan távolodnak el egymástól idővel.

        ---
        ### 📐 Matematikai definíció:
        $$ 
        \\lambda = \\lim_{n \\to \\infty} \\frac{1}{n} \\sum_{i=1}^{n} \\ln \\left| \\frac{df(x_i)}{dx} \\right| 
        $$

        - **λ < 0**: stabil viselkedés (periodikus, fixpont)
        - **λ = 0**: neutrális viselkedés
        - **λ > 0**: káosz, erősen érzékeny rendszer

        ---
        A logisztikus térképen ez a káosz határait és jelenlétét segít feltérképezni, gyakori eszköze a nemlineáris dinamika vizsgálatának.
        """)

# ReflectAI-kompatibilitás
app = run
