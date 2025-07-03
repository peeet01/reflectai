import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==== Dinamikus leképezések ====
def logistic_map(r, x): return r * x * (1 - x)
def quadratic_map(r, x): return r - x ** 2
def henon_map(r, x): return 1 - r * x ** 2  # simplified for 1D use

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
    st.title("🧠 Lyapunov Spektrum – Dinamikus rendszerek stabilitása")

    st.markdown("""
A Lyapunov-exponens azt mutatja meg, mennyire érzékeny egy dinamikus rendszer a kezdeti feltételekre.  
Ez a vizualizáció logisztikus és más típusú leképezések alapján ábrázolja a stabil és kaotikus viselkedés tartományait.
""")

    # Paraméterek
    map_choice = st.selectbox("🧩 Leképezés típusa", list(map_functions.keys()))
    r_min = st.slider("🔽 r minimum érték", 2.5, 3.5, 2.5)
    r_max = st.slider("🔼 r maximum érték", 3.5, 4.0, 4.0)
    n_points = st.slider("📊 Mintapontok száma (r)", 100, 1000, 300, step=50)
    x0 = st.slider("⚙️ Kezdeti érték (x₀)", 0.0, 1.0, 0.5)
    steps = st.slider("🔁 Iterációs lépések száma", 100, 2000, 500, step=100)

    # Spektrum számítása
    r_values = np.linspace(r_min, r_max, n_points)
    map_func = map_functions[map_choice]
    lyap_vals = compute_lyapunov_vectorized(map_func, r_values, x0=x0, steps=steps)

    # === 2D plot ===
    st.subheader("📈 2D Lyapunov-spektrum")
    fig2d, ax = plt.subplots()
    ax.scatter(r_values, lyap_vals, c=np.where(lyap_vals < 0, 'green', 'red'), s=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("r")
    ax.set_ylabel("λ (Lyapunov-exponens)")
    ax.set_title(f"Lyapunov spektrum – {map_choice}")
    st.pyplot(fig2d)

    # === 3D plot ===
    st.subheader("🌐 3D Lyapunov-spektrum")
    R, S = np.meshgrid(r_values, np.arange(steps))
    Z = np.tile(lyap_vals, (steps, 1))
    fig3d = go.Figure(data=[go.Surface(x=R, y=S, z=Z, colorscale="Viridis")])
    fig3d.update_layout(
        title="3D Lyapunov-spektrum",
        scene=dict(xaxis_title='r', yaxis_title='Iteráció', zaxis_title='λ (Lyapunov)'),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # === CSV export ===
    st.subheader("⬇️ Adatok letöltése")
    df = pd.DataFrame({"r": r_values, "lambda": lyap_vals})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV formátumban", data=csv, file_name="lyapunov_spectrum.csv")

    # === Kiértékelés ===
    avg_lyap = np.mean(lyap_vals)
    status = "KAOTIKUS" if avg_lyap > 0 else "STABIL"
    st.success(f"🔍 Az adott beállítások alapján a rendszer **{status}** (átlagos λ = {avg_lyap:.4f})")

    # === Tudományos háttér ===
    st.markdown("### 📚 Tudományos háttér")
    st.markdown(r"""
A **Lyapunov-exponens** egy numerikus mutató, amely azt méri, hogy egy dinamikus rendszer milyen gyorsan tér el a kezdeti feltételektől.  
Ez kulcsfontosságú mutatója a **káosz** jelenlétének.

#### Matematikai definíció:

$$
\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} \ln \left| \frac{df(x_i)}{dx} \right|
$$

#### Értelmezés:

- **λ < 0** → stabil rendszer  
- **λ = 0** → semleges stabilitás  
- **λ > 0** → **káosz**, érzékeny a kezdeti feltételekre

#### Alkalmazás:
- Kaotikus térképek (logisztikus, Henon, kvadratikus)
- Stabilitási régiók feltérképezése
- Nemlineáris dinamikus rendszerek viselkedése
""")

# Kötelező ReflectAI-kompatibilitás
app = run
