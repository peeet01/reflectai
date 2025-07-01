import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def compute_lyapunov_vectorized(r_vals, map_func, x0=0.5, steps=1000, delta=1e-8):
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

# Map functions
def logistic_map(r, x): return r * x * (1 - x)
def tent_map(r, x): return np.where(x < 0.5, r * x, r * (1 - x))
def quadratic_map(r, x): return r - x**2
def henon_map(r, x): return 1 - r * x**2

def run():
    st.title("🧠 Lyapunov Spektrum és Dinamikus Leképezések")

    st.markdown("""
    Ez az alkalmazás különböző **nemlineáris dinamikus leképezések** stabilitását és káoszát vizualizálja a **Lyapunov-exponens** alapján.
    """)

    # Map selection
    map_type = st.selectbox("📊 Leképezés típusa", ["Logisztikus", "Tent", "Quadratic", "Henon"])
    map_dict = {
        "Logisztikus": logistic_map,
        "Tent": tent_map,
        "Quadratic": quadratic_map,
        "Henon": henon_map
    }

    # Parameters
    r_min = st.slider("🔽 r minimum érték", 0.0, 3.9, 2.5)
    r_max = st.slider("🔼 r maximum érték", r_min + 0.1, 4.0, 4.0)
    n_points = st.slider("📊 Mintapontok száma", 100, 2000, 800, step=100)
    x0 = st.slider("⚙️ Kezdeti érték (x₀)", 0.0, 1.0, 0.5)
    steps = st.slider("🔁 Iterációs lépések száma", 100, 3000, 1000, step=100)

    progress = st.progress(0)
    r_values = np.linspace(r_min, r_max, n_points)
    map_func = map_dict[map_type]
    lyapunov_values = compute_lyapunov_vectorized(r_values, map_func, x0=x0, steps=steps)
    progress.progress(100)

    st.subheader("📈 2D Lyapunov-spektrum")
    fig2d, ax = plt.subplots()
    ax.scatter(r_values, lyapunov_values, c=np.where(lyapunov_values < 0, 'green', 'red'), s=2)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("r")
    ax.set_ylabel("λ (Lyapunov-exponens)")
    ax.set_title(f"Lyapunov-spektrum – {map_type} leképezés")
    st.pyplot(fig2d)

    st.subheader("🌐 3D Lyapunov-spektrum")
    R, S = np.meshgrid(r_values, np.arange(steps))
    Z = np.tile(lyapunov_values, (steps, 1))

    fig3d = go.Figure(data=[
        go.Surface(z=Z, x=R, y=S, colorscale="Viridis", showscale=True)
    ])
    fig3d.update_layout(
        title=f"3D Lyapunov-spektrum – {map_type}",
        scene=dict(
            xaxis_title='r paraméter',
            yaxis_title='Iteráció',
            zaxis_title='λ (Lyapunov)',
        ),
        margin=dict(l=0, r=0, t=60, b=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("⬇️ Adatok letöltése")
    df = pd.DataFrame({"r": r_values, "lambda": lyapunov_values})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV formátumban", data=csv, file_name=f"lyapunov_{map_type.lower()}.csv")

    st.subheader("📊 Káosz vagy stabilitás?")
    avg_lyap = np.mean(lyapunov_values)
    if avg_lyap > 0:
        st.success(f"⚠️ Átlagos Lyapunov-exponens: {avg_lyap:.4f} → **Káosz** van jelen a rendszerben!")
    else:
        st.info(f"✅ Átlagos Lyapunov-exponens: {avg_lyap:.4f} → **A rendszer stabil**.")

    with st.expander("📘 Tudományos háttér – Mi az a Lyapunov-exponens?"):
        st.markdown(r"""
        A **Lyapunov-exponens** numerikus mérőszám, amely leírja, hogy egy dinamikus rendszer mennyire érzékeny a kezdeti feltételekre.

        ---
        ### Matematikai definíció:
        $$
        \lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} \ln \left| \frac{df(x_i)}{dx} \right|
        $$

        - **λ < 0** → A rendszer stabil (konvergál)
        - **λ = 0** → Semleges stabilitás
        - **λ > 0** → **Káosz** – az apró eltérések nagy különbségekhez vezetnek idővel

        ---
        A logisztikus, tent, quadratic és Henon leképezések közismert példái a nemlineáris rendszerek kaotikus viselkedésének.
        A Lyapunov-spektrum segít feltárni, hogy mely paraméterek mellett jelenik meg a káosz.
        """)

# ReflectAI-kompatibilitás
app = run
