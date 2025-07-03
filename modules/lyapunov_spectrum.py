"""
Lyapunov Spektrum Modul – Dinamikus rendszerek stabilitásvizsgálata

Ez a modul különböző leképezések mentén számítja és vizualizálja a Lyapunov-exponenseket,
amelyek megmutatják a rendszer érzékenységét a kezdeti feltételekre.

Felhasználási területek:
- Kaotikus viselkedés azonosítása
- Stabilitásvizsgálat
- Dinamikus rendszerek analízise
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==== Dinamikus leképezések ====
def logistic_map(r, x): return r * x * (1 - x)
def quadratic_map(r, x): return r - x ** 2
def henon_map(r, x): return 1 - r * x ** 2  # Egyszerűsített 1D Henon

map_functions = {
    "Logisztikus térkép": logistic_map,
    "Henon térkép": henon_map,
    "Kvadratikus térkép": quadratic_map,
}

# ==== Lyapunov számítás vektorosan ====
def compute_lyapunov_vectorized(f, r_vals, x0=0.5, steps=500, delta=1e-8):
    x = np.full_like(r_vals, x0)
    lyapunov = np.zeros_like(r_vals)
    for _ in range(steps):
        x1 = f(r_vals, x)
        x2 = f(r_vals, x + delta)
        d = np.abs(x2 - x1)
        d = np.where(d == 0, 1e-8, d)
        lyapunov += np.log(np.abs(d / delta))
        x = x1
    return lyapunov / steps

# ==== Streamlit App ====
def run():
    st.title("📊 Lyapunov Spektrum – Dinamikus rendszerek stabilitása")

    st.markdown("""
A **Lyapunov-exponens** egy numerikus mérőszám, amely azt jelzi, hogy egy dinamikus rendszer
mennyire érzékeny a kezdeti feltételekre. E modul segítségével kiszámíthatjuk a logisztikus és más leképezések spektrumát.
""")

    # 🎛️ Paraméterek beállítása
    st.sidebar.header("⚙️ Paraméterek")
    map_choice = st.sidebar.selectbox("🧩 Leképezés típusa", list(map_functions.keys()))
    r_min = st.sidebar.slider("🔽 r minimum érték", 2.5, 3.5, 2.5)
    r_max = st.sidebar.slider("🔼 r maximum érték", 3.5, 4.0, 4.0)
    n_points = st.sidebar.slider("📊 Mintapontok száma (r)", 100, 1000, 300, step=50)
    x0 = st.sidebar.slider("⚙️ Kezdeti érték (x₀)", 0.0, 1.0, 0.5)
    steps = st.sidebar.slider("🔁 Iterációs lépések száma", 100, 2000, 500, step=100)

    # 📈 Számítás
    r_values = np.linspace(r_min, r_max, n_points)
    map_func = map_functions[map_choice]
    lyap_vals = compute_lyapunov_vectorized(map_func, r_values, x0=x0, steps=steps)

    # === 2D ÁBRA ===
    st.subheader("📈 2D Lyapunov-spektrum")
    fig2d, ax = plt.subplots()
    ax.scatter(r_values, lyap_vals, c=np.where(lyap_vals < 0, 'green', 'red'), s=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("r")
    ax.set_ylabel("λ (Lyapunov-exponens)")
    ax.set_title(f"Lyapunov spektrum – {map_choice}")
    st.pyplot(fig2d)

    # === 3D ÁBRA ===
    st.subheader("🌐 3D Lyapunov-spektrum")
    R, S = np.meshgrid(r_values, np.arange(steps))
    Z = np.tile(lyap_vals, (steps, 1))
    fig3d = go.Figure(data=[go.Surface(x=R, y=S, z=Z, colorscale="Viridis")])
    fig3d.update_layout(
        title="3D Lyapunov-spektrum",
        scene=dict(
            xaxis_title='r',
            yaxis_title='Iteráció',
            zaxis_title='λ (Lyapunov)'
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # === CSV export ===
    st.subheader("⬇️ Eredmények exportálása")
    df = pd.DataFrame({"r": r_values, "lambda": lyap_vals})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV formátumban", data=csv, file_name="lyapunov_spectrum.csv")

    # === Átlagérték és értelmezés ===
    avg_lyap = np.mean(lyap_vals)
    status = "KAOTIKUS" if avg_lyap > 0 else "STABIL"
    st.success(f"🔍 A rendszer viselkedése: **{status}** (átlagos λ = {avg_lyap:.4f})")

    # === Tudományos háttér ===
    with st.expander("📘 Tudományos háttér – Lyapunov-exponens"):
        st.markdown(r"""
A **Lyapunov-exponens** egy dinamikus rendszerben a kezdeti feltételek perturbációira adott válasz mérőszáma.

### 📐 Matematikai definíció:

$$
\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} \ln \left| \frac{df(x_i)}{dx} \right|
$$

Ahol:
- \( \lambda \) a Lyapunov-exponens
- \( f(x) \) a leképezés
- \( x_i \) az aktuális állapot

### 📊 Értelmezés:
- **λ < 0** → stabil rendszer (konvergál)
- **λ = 0** → semleges stabilitás
- **λ > 0** → **káosz** – extrém érzékenység a kezdeti értékekre

### 📌 Tipikus alkalmazások:
- Kaotikus rendszerek jellemzése
- Idősorok stabilitásvizsgálata
- Biológiai és ökológiai modellek dinamikája
""")

# ReflectAI kompatibilitás
app = run
