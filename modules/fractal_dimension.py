"""
Fractal Dimension Module – Box Counting (javított, fixált skála).

Fő különbség: a pontfelhőt egyszer normalizáljuk egy [0,1]^2 négyzetre,
és minden ε méretezést ezen a FIX rácson végzünk. Így a skálázási törvény
nem torzul az egyes ε értékeknél újramin-maxolástól.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

# -------------------- Segédfüggvények --------------------

def normalize_unit_square(data, eps_clip=1e-12):
    """
    Adathalmazt egy, az adatra feszített NÉGYZETRE (nem csak téglalapra) skáláz,
    majd [0,1] intervallumba visz mindkét tengelyen. A széleknél kissé beljebb
    klippel, hogy a floor() indexelés ne ugorjon ki.
    """
    data = np.asarray(data, dtype=np.float64)
    mn = data.min(axis=0)
    mx = data.max(axis=0)
    ctr = (mn + mx) / 2.0
    span = (mx - mn).max()  # négyzet oldala
    # ha degenerált (egy pont), adjunk kis span-t
    if span == 0:
        span = 1.0
    data_sq = (data - (ctr - span/2)) / span  # [0,1]x[0,1] köré
    # numerikus stabilitás – kicsit beljebb klippelünk
    data_sq = np.clip(data_sq, eps_clip, 1.0 - eps_clip)
    return data_sq

def histogram2d_density(data, bins=100):
    H, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=bins, range=[[0,1],[0,1]])
    Z = H.T  # plotly miatt
    return Z

def box_count_fixed_grid(data01, epsilons):
    """
    FIX [0,1]^2 rácson számolja N(ε)-t.
    Indexelés: i = floor(x/ε), j = floor(y/ε)
    """
    counts = []
    for eps in epsilons:
        # dobozok száma oldalanként: ~ floor(1/eps)
        # indexek:
        ij = np.floor(data01 / eps).astype(np.int64)
        # védekezés az eps túl nagy/túl kicsi esetre:
        if ij.size == 0:
            counts.append(0)
        else:
            # egyedi dobozok
            counts.append(len(np.unique(ij, axis=0)))
    return np.array(counts, dtype=np.int64)

def fit_loglog(log_inv_eps, logN):
    # egyszerű OLS + R^2
    A = np.vstack([log_inv_eps, np.ones_like(log_inv_eps)]).T
    slope, intercept = np.linalg.lstsq(A, logN, rcond=None)[0]
    yhat = slope*log_inv_eps + intercept
    ss_res = np.sum((logN - yhat)**2)
    ss_tot = np.sum((logN - logN.mean())**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2

# -------------------- Mintagenerátorok --------------------

def gen_spiral(n):
    theta = np.linspace(0, 4*np.pi, n)
    r = np.linspace(0.1, 1.0, n)
    x = r*np.cos(theta); y = r*np.sin(theta)
    return np.c_[x, y]

def gen_lorenz_project(n, dt=0.01, a=10.0, b=28.0, c=8/3):
    x, y, z = 0.0, 1.0, 1.05
    X, Y = [x], [y]
    for _ in range(n-1):
        dx = a * (y - x)
        dy = x * (b - z) - y
        dz = x * y - c * z
        x += dx * dt; y += dy * dt; z += dz * dt
        X.append(x); Y.append(y)
    return np.c_[X, Y]

def gen_random(n):
    return np.random.rand(n, 2)

def gen_sierpinski(n, steps=50000, seed=42):
    """
    Káosz-játék (IFS) Sierpinski-háromszög – jó benchmark (D = ln3/ln2 ~ 1.585).
    Az első néhány ezer iterációt burn-in-nek tekintjük.
    """
    rng = np.random.default_rng(seed)
    V = np.array([[0,0],[1,0],[0.5, np.sqrt(3)/2]], dtype=np.float64)
    x = rng.random(2)
    pts = []
    burn = min(steps//10, 2000)
    for i in range(steps):
        v = V[rng.integers(0,3)]
        x = (x + v)/2.0
        if i >= burn:
            pts.append(x.copy())
            if len(pts) >= n:
                break
    if len(pts) < n:
        pts.extend([x.copy()]*(n-len(pts)))
    return np.array(pts)

CLOUDS = {
    "Szimmetrikus spirál": gen_spiral,
    "Lorenz-projekció": gen_lorenz_project,
    "Random felhő": gen_random,
    "Sierpinski-háromszög (IFS)": gen_sierpinski,
}

EXPECTED_D = {
    "Sierpinski-háromszög (IFS)": np.log(3)/np.log(2),  # ~1.585
    # a többinél nincs kanonikus "igazi" érték 2D projekcióra
}

# -------------------- Streamlit App --------------------

def run():
    st.set_page_config(layout="wide")
    st.title("🧮 Fraktál dimenzió – Box counting (fixált skálán)")

    st.markdown(r"""
A fraktál dimenzió **Minkowski–Bouligand** (box-counting) definíciója:
\[
D \;=\; \lim_{\varepsilon\to 0} \frac{\log N(\varepsilon)}{\log(1/\varepsilon)}\,,
\]
ahol \(N(\varepsilon)\) az \(\varepsilon\) oldalú dobozok minimális száma,
amely lefedi a mintázatot. A becsléshez \(\log N(\varepsilon)\)–\(\log(1/\varepsilon)\) egyenest illesztünk.
""")

    # Paraméterek
    st.sidebar.header("⚙️ Paraméterek")
    kind = st.sidebar.selectbox("🔘 Pontfelhő típusa", list(CLOUDS.keys()))
    n_points = st.sidebar.slider("📊 Pontok száma", 200, 50000, 3000, step=100)
    noise_pct = st.sidebar.slider("📉 Zajszint (%)", 0, 50, 0, step=1)
    grid_3d = st.sidebar.slider("🧱 3D rács (bins)", 20, 200, 100, step=10)

    # ε tartomány log-skálán (stabil)
    eps_start = st.sidebar.slider("📏 log10 ε (kezdő)", -2.5, -0.3, -2.0, 0.1)
    eps_end   = st.sidebar.slider("📏 log10 ε (vég)",   -2.0,  0.0, -0.6, 0.1)
    n_scales  = st.sidebar.slider("📈 Skálák száma", 6, 20, 10)

    # generálás + normalizálás
    gen_fn = CLOUDS[kind]
    data = gen_fn(n_points).astype(np.float64)
    if noise_pct > 0:
        data += (noise_pct/100.0) * np.random.randn(*data.shape)
    data01 = normalize_unit_square(data)

    # ε-k előkészítése (helyes irány, klippelés)
    lo, hi = (eps_start, eps_end) if eps_start < eps_end else (eps_end, eps_start)
    epsilons = np.logspace(lo, hi, n_scales)  # természetes egységben (unit square)

    # box counting
    counts = box_count_fixed_grid(data01, epsilons)

    # szűrés: csak pozitív N(ε)
    msk = counts > 0
    log_inv_eps = np.log(1.0/epsilons[msk])
    logN = np.log(counts[msk])

    slope, intercept, r2 = fit_loglog(log_inv_eps, logN)
    D_hat = slope

    colA, colB = st.columns([1,1])

    with colA:
        st.subheader("🌀 Pontfelhő (normalizálva [0,1]²-re)")
        fig1, ax1 = plt.subplots()
        ax1.scatter(data01[:,0], data01[:,1], s=2, alpha=0.7)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlim(0,1); ax1.set_ylim(0,1)
        ax1.set_xticks([]); ax1.set_yticks([])
        st.pyplot(fig1)

        st.subheader("🧱 2D occupancy heatmap")
        H = histogram2d_density(data01, bins=grid_3d)
        fighm, axhm = plt.subplots()
        axhm.imshow(H, origin="lower", cmap="inferno", interpolation="nearest")
        axhm.set_xticks([]); axhm.set_yticks([])
        st.pyplot(fighm)

    with colB:
        st.subheader("📐 Box counting – log–log illesztés")
        fig2, ax2 = plt.subplots()
        ax2.plot(log_inv_eps, logN, "o", label="Mért pontok")
        xfit = np.linspace(log_inv_eps.min(), log_inv_eps.max(), 100)
        ax2.plot(xfit, slope*xfit + intercept, "--", label=f"Illesztés: D ≈ {D_hat:.3f}")
        ax2.set_xlabel(r"$\log(1/\varepsilon)$")
        ax2.set_ylabel(r"$\log N(\varepsilon)$")
        ax2.legend()
        st.pyplot(fig2)

        # státusz
        expD = EXPECTED_D.get(kind, None)
        if expD is not None:
            st.info(f"🔎 **Becsült D:** {D_hat:.3f}  |  **Elméleti:** {expD:.3f}  |  **R²:** {r2:.3f}")
        else:
            st.info(f"🔎 **Becsült D:** {D_hat:.3f}  |  **R²:** {r2:.3f}")

    # 3D sűrűség
    st.subheader("🌋 3D sűrűségfelület")
    Z = histogram2d_density(data01, bins=grid_3d)
    xx = np.arange(Z.shape[1]); yy = np.arange(Z.shape[0])
    X, Y = np.meshgrid(xx, yy)
    fig3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Inferno")])
    fig3d.update_layout(
        title="3D fraktál sűrűség",
        margin=dict(l=0, r=0, b=0, t=40),
        height=520
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # Export
    st.subheader("📥 CSV export")
    out = pd.DataFrame({
        "epsilon": epsilons[msk],
        "N_eps": counts[msk],
        "log_1_over_eps": log_inv_eps,
        "log_N": logN
    })
    st.download_button("⬇️ Box-counting eredmények (CSV)",
                       data=out.to_csv(index=False).encode("utf-8"),
                       file_name="box_counting_results.csv",
                       mime="text/csv")

    st.markdown("### 📚 Tudományos megjegyzések")
    st.markdown(r"""
- A pontfelhőt **egyszer** normalizáljuk egy **fix [0,1]^2 négyzetre** → a skálázás konzisztens.
- A dimenzió \(D\) a lejtő az \(\log N(\varepsilon)\) – \(\log (1/\varepsilon)\) grafikonon.
- Az **R²** a lineáris skálázás minőségét jelzi; az érvényes \(\varepsilon\)-tartomány kiválasztása kritikus.
- Benchmark: a *Sierpinski-háromszög* esetén \(D=\ln 3/\ln 2 \approx 1.585\).
""")

# ReflectAI kompat
app = run
