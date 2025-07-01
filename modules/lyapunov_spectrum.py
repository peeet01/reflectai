import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# === MAP DEFINÍCIÓK ===
def logistic_map(r, x): return r * x * (1 - x)
def tent_map(r, x): return r * min(x, 1 - x)
def quadratic_map(r, x): return np.clip(r - x**2, -2, 2)
def henon_map(r, x): return np.clip(1 - r * x**2, -2, 2)

map_functions = {
    "Logisztikus": logistic_map,
    "Quadratic": quadratic_map,
    "Tent": tent_map,
    "Henon-szerű": henon_map,
}

# === LYAPUNOV ===
def compute_lyapunov_vectorized(f, r_vals, x0=0.5, steps=300, delta=1e-8):
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

# === APP ===
def run():
    st.title("🧠 Lyapunov Spektrum – Dinamikus rendszerek stabilitása")

    st.markdown("""
    Vizualizáció különböző leképezések alapján. A **Lyapunov-exponens** segít megérteni, mikor válik egy rendszer kaotikussá a paraméterfüggés szerint.
    """)

    map_choice = st.selectbox("📌 Leképezés típusa", list(map_functions.keys()))
    r_min = st.slider("🔽 r minimum", 2.5, 3.5, 2.5)
    r_max = st.slider("🔼 r maximum", 3.5, 4.0, 4.0)
    n_points = st.slider("📊 Mintapontok száma", 100, 1000, 500, step=100)
    steps = st.slider("🔁 Iterációk száma", 100, 1000, 300, step=100)
    x0 = st.slider("⚙️ Kezdeti érték (x₀)", 0.0, 1.0, 0.5)

    r_values = np.linspace(r_min, r_max, n_points)
    f_map = map_functions[map_choice]
    lyap_vals = compute_lyapunov_vectorized(f_map, r_values, x0=x0, steps=steps)

    # === 2D ÁBRA ===
    st.subheader("📈 2D Lyapunov-spektrum")
    fig2d, ax = plt.subplots()
    ax.scatter(r_values, lyap_vals, c=np.where(lyap_vals < 0, 'green', 'red'), s=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("r")
    ax.set_ylabel("λ (Lyapunov-exponens)")
    ax.set_title(f"Lyapunov spektrum – {map_choice} leképezés")
    st.pyplot(fig2d)

    # === 3D ÁBRA ===
    st.subheader("🌐 3D Lyapunov-spektrum")
    x3d, y3d = np.meshgrid(r_values, np.arange(steps))
    z3d = np.array([lyap_vals for _ in range(steps)])
    fig3d = go.Figure(data=[go.Surface(z=z3d, x=x3d, y=y3d, colorscale='Inferno')])
    fig3d.update_layout(
        title=f"3D Lyapunov-spektrum – {map_choice}",
        scene=dict(
            xaxis_title='r',
            yaxis_title='Iteráció',
            zaxis_title='λ (Lyapunov)',
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # === KIÉRTÉKELÉS ===
    st.subheader("🧪 Rendszer stabilitása")
    kaotikus_arany = np.sum(lyap_vals > 0) / len(lyap_vals)
    if kaotikus_arany > 0.5:
        st.error(f"A rendszer **többségében kaotikus** ezen a tartományon ({kaotikus_arany:.1%}).")
    elif kaotikus_arany > 0:
        st.warning(f"**Vegyes** viselkedés: stabil és kaotikus zónák is előfordulnak ({kaotikus_arany:.1%}).")
    else:
        st.success("A rendszer **stabil** ezen a tartományon – nincs pozitív Lyapunov-exponens.")

    # === CSV EXPORT ===
    st.subheader("⬇️ Adatok letöltése")
    df = pd.DataFrame({"r": r_values, "lambda": lyap_vals})
    st.download_button("Letöltés CSV-ben", data=df.to_csv(index=False).encode('utf-8'), file_name="lyapunov_spectrum.csv")

    # === TUDOMÁNYOS MAGYARÁZAT ===
    with st.expander("📘 Tudományos háttér – Mi az a Lyapunov-exponens?"):
        st.markdown("""
        A **Lyapunov-exponens** numerikus mérőszám, amely azt mutatja meg, hogy egy dinamikus rendszer mennyire érzékeny a kezdeti feltételekre.

        ---
        **Matematikai definíció**:
        $$
        \\lambda = \\lim_{n \\to \\infty} \\frac{1}{n} \\sum_{i=1}^{n} \\ln \\left| \\frac{df(x_i)}{dx} \\right|
        $$

        - Ha **λ < 0** → stabil (fixpont vagy ciklus)
        - Ha **λ = 0** → neutrális stabilitás
        - Ha **λ > 0** → **káosz**: kis eltérés is drasztikus következményekkel jár

        A logisztikus, tent, quadratic vagy Henon típusú leképezések mind kiváló példái a nemlineáris rendszerek viselkedésére.
        A Lyapunov-spektrum feltárja, hogy a paraméterek változása hogyan befolyásolja a rendszer dinamikáját.
        """)

# ReflectAI-kompatibilitás
app = run
