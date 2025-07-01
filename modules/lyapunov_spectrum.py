import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Gyorsított Lyapunov-exponens számítás
def compute_lyapunov_vectorized(r_vals, x0=0.5, steps=1000, delta=1e-8):
    x = np.full_like(r_vals, x0)
    lyapunov = np.zeros_like(r_vals)

    for _ in range(steps):
        x1 = r_vals * x * (1 - x)
        x2 = r_vals * (x + delta) * (1 - (x + delta))
        d = np.abs(x2 - x1)
        d = np.where(d == 0, 1e-8, d)
        lyapunov += np.log(np.abs(d / delta))
        x = x1
    return lyapunov / steps

def run():
    st.title("🧠 Lyapunov Spektrum – Dinamikus rendszerek stabilitása")

    st.markdown("""
    Vizualizáció logisztikus leképezés alapján, amely a káosz határát mutatja meg a Lyapunov-exponens segítségével.
    """)

    # Paraméterek
    r_min = st.slider("🔽 r minimum érték", 2.5, 3.5, 2.5)
    r_max = st.slider("🔼 r maximum érték", 3.5, 4.0, 4.0)
    n_points = st.slider("📊 r pontok száma", 100, 2000, 800, step=100)
    x0 = st.slider("⚙️ Kezdeti érték (x₀)", 0.0, 1.0, 0.5)
    steps = st.slider("🔁 Iterációs lépések száma", 100, 3000, 1000, step=100)

    # Spektrum számítása
    r_values = np.linspace(r_min, r_max, n_points)
    lyapunov_values = compute_lyapunov_vectorized(r_values, x0=x0, steps=steps)

    # 2D plot
    st.subheader("📈 2D Lyapunov-spektrum")
    fig2d, ax = plt.subplots()
    ax.scatter(r_values, lyapunov_values, c=np.where(lyapunov_values < 0, 'green', 'red'), s=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("r")
    ax.set_ylabel("λ (Lyapunov-exponens)")
    ax.set_title("Lyapunov spektrum – logisztikus térkép")
    st.pyplot(fig2d)

    # 3D plot – ÚJ, színes és gyors plotly-s verzió
    st.subheader("🌐 3D Lyapunov-spektrum")

    x0_min = st.slider("⚙️ x₀ minimum", 0.0, 1.0, 0.1)
    x0_max = st.slider("⚙️ x₀ maximum", 0.0, 1.0, 0.9)
    x0_points = st.slider("📈 x₀ pontok száma", 10, 100, 40)

    x0_values = np.linspace(x0_min, x0_max, x0_points)
    R_grid, X0_grid = np.meshgrid(r_values, x0_values)
    Z_grid = np.zeros_like(R_grid)

    for i in range(x0_points):
        lyaps = compute_lyapunov_vectorized(r_values, x0=x0_values[i], steps=steps)
        Z_grid[i, :] = lyaps

    fig3d = go.Figure(data=[go.Surface(
        x=R_grid,
        y=X0_grid,
        z=Z_grid,
        colorscale='Viridis',
        showscale=True
    )])

    fig3d.update_layout(
        title="3D Lyapunov-spektrum (r × x₀ tér)",
        scene=dict(
            xaxis_title='r',
            yaxis_title='x₀',
            zaxis_title='λ (Lyapunov)',
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # CSV export
    st.subheader("⬇️ Adatok letöltése")
    df = pd.DataFrame({"r": r_values, "lambda": lyapunov_values})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV formátumban", data=csv, file_name="lyapunov_spectrum.csv")

    # Tudományos háttér
    with st.expander("📘 Tudományos háttér – Mi az a Lyapunov-exponens?"):
        st.markdown("""
        A **Lyapunov-exponens** numerikus mérőszám, amely azt mutatja meg, hogy egy dinamikus rendszer mennyire érzékeny a kezdeti feltételekre.

        ---
        **Matematikai definíció**:
        $$
        \\lambda = \\lim_{n \\to \\infty} \\frac{1}{n} \\sum_{i=1}^{n} \\ln \\left| \\frac{df(x_i)}{dx} \\right|
        $$

        - Ha **λ < 0**: stabil rendszer (konvergál)
        - Ha **λ = 0**: semleges stabilitás
        - Ha **λ > 0**: **káosz** – kis eltérés is drasztikus kimenethez vezet

        A logisztikus leképezés klasszikus példája ennek a viselkedésnek. A Lyapunov-spektrum pedig a stabil és kaotikus zónákat tárja fel.
        """)

# ReflectAI-kompatibilitás
app = run
