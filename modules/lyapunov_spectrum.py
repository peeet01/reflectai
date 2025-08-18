import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ==== 1D dinamikus leképezések és analitikus deriváltak ====
def logistic_map(r, x):      # x_{n+1} = r x (1-x), r∈[0,4]
    return r * x * (1 - x)
def d_logistic_map(r, x):    # f'(x) = r (1 - 2x)
    return r * (1 - 2*x)

def quadratic_map(r, x):     # x_{n+1} = r - x^2  (klasszikus kvadratikus család)
    return r - x**2
def d_quadratic_map(r, x):   # f'(x) = -2x
    return -2*x

# FIGYELEM: a "Henon" 1D változat nem a valódi 2D Hénon-térkép!
def pseudo_henon_map(r, x):  # x_{n+1} = 1 - r x^2 (csak demonstráció)
    return 1 - r * x**2
def d_pseudo_henon_map(r, x):
    return -2*r*x

map_functions = {
    "Logisztikus térkép": (logistic_map, d_logistic_map),
    "Kvadratikus térkép": (quadratic_map, d_quadratic_map),
    "Pseudo-Hénon (1D)": (pseudo_henon_map, d_pseudo_henon_map),
}

# ==== Lyapunov számítás analitikus deriválttal + burn-in ====
def lyapunov_spectrum_1d(map_f, dmap_f, r_vals, x0=0.5, steps=1500, burn_in=500):
    """
    λ(r) = lim (1/N) Σ ln | f'(x_n; r) |  a burn-in utáni lépésekre.
    Minden r-hez külön trajektória fut (vektorosítva).
    """
    r_vals = np.asarray(r_vals, dtype=np.float64)
    x = np.full_like(r_vals, fill_value=x0, dtype=np.float64)

    # Burn-in: csak evolválunk, nem átlagolunk
    for _ in range(burn_in):
        x = map_f(r_vals, x)

    # Fő ciklus: átlag a burn-in utáni lépésekre
    lyap_sum = np.zeros_like(r_vals, dtype=np.float64)
    lyap_history = []  # (iter, r) mátrix a 3D felülethez

    for n in range(1, steps + 1):
        # analitikus derivált – numerikus stabilitás
        deriv = np.abs(dmap_f(r_vals, x))
        deriv = np.clip(deriv, 1e-300, None)  # elkerüljük a log(0)-t
        lyap_sum += np.log(deriv)
        lyap_history.append(lyap_sum / n)
        x = map_f(r_vals, x)

    lyap_vals = lyap_sum / steps
    lyap_history = np.vstack(lyap_history)    # shape: (steps, len(r_vals))
    return lyap_vals, lyap_history

# ==== Opcionális: véges differenciás közelítés ====
def lyapunov_finite_diff(map_f, r_vals, x0=0.5, steps=1500, burn_in=500, delta=1e-8):
    r_vals = np.asarray(r_vals, dtype=np.float64)
    x = np.full_like(r_vals, fill_value=x0, dtype=np.float64)

    for _ in range(burn_in):
        x = map_f(r_vals, x)

    lyap_sum = np.zeros_like(r_vals, dtype=np.float64)
    lyap_history = []

    for n in range(1, steps + 1):
        x1 = map_f(r_vals, x)
        x2 = map_f(r_vals, x + delta)
        d = np.abs(x2 - x1) / delta
        d = np.clip(d, 1e-300, None)
        lyap_sum += np.log(d)
        lyap_history.append(lyap_sum / n)
        x = x1

    return lyap_sum / steps, np.vstack(lyap_history)

# ==== Streamlit App ====
def run():
    st.title("🧠 Lyapunov Spektrum – Dinamikus rendszerek stabilitása")

    st.markdown("""
A Lyapunov-exponens a **kezdeti feltételekre való érzékenységet** méri diszkrét leképezéseknél:  
\\[
\\lambda = \\lim_{n\\to\\infty} \\frac{1}{n} \\sum_{i=1}^{n} \\ln\\left|f'(x_i)\\right|.
\\]
Pozitív \\(\\lambda\\) → **káosz**, negatív → **stabil** (attraktorba húz).
""")

    # Paraméterek
    map_choice = st.selectbox("🧩 Leképezés típusa", list(map_functions.keys()))
    r_min, r_max = st.slider("r tartomány", 0.0, 4.0, (2.5, 4.0))
    n_points = st.slider("📊 Mintapontok száma (r)", 100, 2000, 600, step=100)
    x0 = st.slider("⚙️ Kezdeti érték (x₀)", 0.0, 1.0, 0.5)
    burn_in = st.slider("🔥 Burn-in lépések", 0, 5000, 500, step=100)
    steps = st.slider("🔁 Átlagolt lépések száma", 100, 5000, 1000, step=100)

    use_finite_diff = st.checkbox("Nincs analitikus derivált – használj véges differenciát", value=False)

    # Spektrum számítása
    r_values = np.linspace(r_min, r_max, n_points)
    map_f, dmap_f = map_functions[map_choice]

    if use_finite_diff:
        lyap_vals, lyap_hist = lyapunov_finite_diff(
            map_f, r_values, x0=x0, steps=steps, burn_in=burn_in
        )
    else:
        lyap_vals, lyap_hist = lyapunov_spectrum_1d(
            map_f, dmap_f, r_values, x0=x0, steps=steps, burn_in=burn_in
        )

    # === 2D plot ===
    st.subheader("📈 2D Lyapunov-spektrum")
    fig2d, ax = plt.subplots()
    ax.scatter(r_values, lyap_vals, c=np.where(lyap_vals < 0, 'green', 'red'), s=6, alpha=0.9)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("r")
    ax.set_ylabel("λ (Lyapunov-exponens)")
    ax.set_title(f"Lyapunov-spektrum – {map_choice}")
    st.pyplot(fig2d)

    # === 3D plot: a részátlag konvergenciája (ROBOSZTUS) ===
    st.subheader("🌐 3D – Konvergencia az iterációk mentén")
    R, Ngrid = np.meshgrid(r_values, np.arange(1, steps+1))

    # Biztonság: csak véges értékekkel dolgozunk
    Z = np.array(lyap_hist, dtype=np.float64)
    Z[~np.isfinite(Z)] = np.nan

    finite = Z[np.isfinite(Z)]
    if finite.size == 0 or (np.nanmax(Z) - np.nanmin(Z) < 1e-9):
        # Fallback: 2D hőtérkép, ha túl lapos vagy minden NaN
        st.warning("A 3D felület túl lapos vagy nem véges értékeket tartalmaz – 2D hőtérkép jelenik meg.")
        fighm = go.Figure(data=[go.Heatmap(x=r_values, y=np.arange(1, steps+1), z=Z)])
        fighm.update_layout(
            xaxis_title='r', yaxis_title='n (iteráció)', title='Részátlagolt λ – hőtérkép',
            margin=dict(l=0, r=0, t=60, b=0), height=520
        )
        st.plotly_chart(fighm, use_container_width=True)
    else:
        # Dinamikus színtartomány – szélsőségek levágása, hogy ne „feketedjen” a felület
        cmin = float(np.nanpercentile(Z, 2))
        cmax = float(np.nanpercentile(Z, 98))
        if cmax - cmin < 1e-9:  # extra biztosíték
            cmin, cmax = float(np.nanmin(Z)), float(np.nanmax(Z))

        fig3d = go.Figure(data=[go.Surface(
            x=R, y=Ngrid, z=Z, colorscale="Viridis", cmin=cmin, cmax=cmax, showscale=True
        )])
        fig3d.update_layout(
            title="Részátlagolt λ(r, n) felület",
            scene=dict(xaxis_title='r', yaxis_title='n (iteráció)', zaxis_title='λ részátlag'),
            margin=dict(l=0, r=0, t=60, b=0),
            height=520
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # === CSV export ===
    st.subheader("⬇️ Adatok letöltése")
    df = pd.DataFrame({"r": r_values, "lambda": lyap_vals})
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Letöltés CSV formátumban", data=csv, file_name="lyapunov_spectrum.csv")

    # === Rövid diagnózis ===
    frac_chaotic = np.mean(lyap_vals > 0)
    st.info(f"🔍 A mintavételezett r-tartomány {frac_chaotic*100:.1f}%-ában λ>0 (káosz).")

    st.subheader("📚 Tudományos háttér")

st.latex(r"\lambda = \lim_{n\to\infty}\frac{1}{n}\sum_{i=1}^{n}\ln\!\left|f'(x_i)\right|,\qquad x_{i+1}=f(x_i).")

st.latex(r"\text{Logisztikus térkép: }\; x_{n+1}=r x_n(1-x_n),\quad f'(x)=r(1-2x)")
st.latex(r"\text{Kvadratikus térkép: }\; x_{n+1}=r-x_n^2,\quad f'(x)=-2x")

st.markdown(
    "A **burn-in** eltávolítja a kezdeti transzienseket; "
    "a részátlag \\(\\lambda_n\\) konvergenciáját mutatja a 3D felület."
)

# Kötelező ReflectAI-kompatibilitás
app = run
