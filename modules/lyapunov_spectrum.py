import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# === Leképezések definíciói ===
def logistic_map(r, x): return r * x * (1 - x)
def tent_map(r, x): return r * np.minimum(x, 1 - x)
def quadratic_map(r, x): return r - x**2
def henon_map(a, b, x, y): return 1 - a * x**2 + y, b * x

# === Gyorsított Lyapunov számítás 1D leképezésekhez ===
def compute_lyapunov_vectorized_1d(map_func, r_vals, x0=0.5, steps=1000, delta=1e-8):
    x = np.full_like(r_vals, x0)
    lyapunov = np.zeros_like(r_vals)

    for _ in range(steps):
        x1 = map_func(r_vals, x)
        x2 = map_func(r_vals, x + delta)
        d = np.abs(x2 - x1)
        d = np.where(d == 0, 1e-8, d)
        lyapunov += np.log(np.abs(d / delta))
        x = x1
    return lyapunov / steps

# === Streamlit App ===
def run():
    st.title("🧠 Többtérképes Lyapunov Spektrum")
    st.markdown("""
    Ez a modul lehetőséget ad különböző diszkrét dinamikus rendszerek (logisztikus, tent, kvadratikus és Henon) Lyapunov-spektrumának kiszámítására és vizualizálására.
    """)

    # --- Leképezés kiválasztása ---
    map_choice = st.selectbox("🧮 Leképezés típusa", ["Logisztikus", "Tent", "Kvadratikus"])
    r_min = st.slider("🔽 r minimum érték", 0.0, 4.0, 2.5)
    r_max = st.slider("🔼 r maximum érték", 0.1, 4.0, 4.0)
    n_points = st.slider("📊 Mintapontok száma", 100, 2000, 800, step=100)
    x0 = st.slider("⚙️ Kezdeti érték (x₀)", 0.0, 1.0, 0.5)
    steps = st.slider("🔁 Iterációs lépések száma", 100, 3000, 1000, step=100)

    # --- Leképezés hozzárendelés ---
    map_func = {
        "Logisztikus": logistic_map,
        "Tent": tent_map,
        "Kvadratikus": quadratic_map
    }[map_choice]

    r_values = np.linspace(r_min, r_max, n_points)
    lyapunov_values = compute_lyapunov_vectorized_1d(map_func, r_values, x0=x0, steps=steps)

    # --- 2D Plot ---
    st.subheader("📈 2D Lyapunov-spektrum")
    fig2d, ax = plt.subplots()
    ax.scatter(r_values, lyapunov_values, c=np.where(lyapunov_values < 0, 'green', 'red'), s=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("r")
    ax.set_ylabel("λ (Lyapunov-exponens)")
    ax.set_title(f"Lyapunov spektrum – {map_choice} leképezés")
    st.pyplot(fig2d)

    # --- 3D Plot ---
    st.subheader("🌐 3D Lyapunov-spektrum")
    R, S = np.meshgrid(r_values, np.arange(steps))
    Z = np.tile(lyapunov_values, (steps, 1))
    fig3d = go.Figure(data=[go.Surface(x=R, y=S, z=Z, colorscale="Inferno", showscale=False)])
    fig3d.update_layout(
        scene=dict(xaxis_title="r", yaxis_title="Iteráció", zaxis_title="λ"),
        margin=dict(l=0, r=0, t=40, b=0),
        title="3D Lyapunov spektrum"
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # --- CSV Export ---
    st.subheader("⬇️ Adatok letöltése")
    df = pd.DataFrame({"r": r_values, "lambda": lyapunov_values})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV formátumban", data=csv, file_name="lyapunov_spectrum.csv")

    # --- Tudományos háttér ---
    with st.expander("📘 Tudományos háttér – Mi az a Lyapunov-exponens?"):
        st.markdown(f"""
        A **Lyapunov-exponens** egy kulcsfontosságú mérőszám, amely azt vizsgálja, hogy egy rendszer **mennyire érzékeny a kezdeti feltételekre**.  
        Különböző leképezések eltérő dinamikát mutatnak:
        - **Logisztikus**: klasszikus bifurkációs és káotikus viselkedés.
        - **Tent**: darabos, de jól kontrollálható káosz.
        - **Kvadratikus**: nemlineáris inverziókat tartalmaz.

        ### Matematikai meghatározás:
        $$
        \\lambda = \\lim_{{n \\to \\infty}} \\frac{{1}}{{n}} \\sum_{{i=1}}^n \\ln \\left| \\frac{{df(x_i)}}{{dx}} \\right|
        $$

        - Ha **λ < 0** → stabil rendszer  
        - Ha **λ > 0** → **káosz** – az eltérések exponenciálisan nőnek  
        - **λ = 0** → neutrális viselkedés

        A fenti ábrák segítenek feltérképezni a **kaotikus zónák** elhelyezkedését a paramétertérben.
        """)

# ReflectAI-kompatibilitás
app = run
