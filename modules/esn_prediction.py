import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ------- Bemeneti jel generálás -------
def gen_input_signal(kind, T, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 50, T)
    if kind == "Szinusz":
        u = np.sin(0.4 * t) + 0.3*np.sin(0.05 * t + 0.7)
    elif kind == "Mackey–Glass (szintetikus)":
        # kvázi-kaotikus kevert frekvenciák + kicsi zaj
        u = np.sin(0.2*t) * np.sin(0.311*t + 1.2) + 0.1*rng.standard_normal(T)
    else:
        u = np.sin(0.35 * t)
    return u

# ------- Spektrálsugár skálázás (gyors power-iteráció) -------
def power_iteration_spectral_radius(W, iters=20):
    # becsli a legnagyobb sajátérték abszolút értékét
    n = W.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(iters):
        v = W @ v
        nrm = np.linalg.norm(v) + 1e-12
        v /= nrm
    # Rayleigh-hányados
    lam = v @ (W @ v)
    return float(abs(lam))

def scale_spectral_radius(W, rho_target=0.9, iters=20):
    sr = power_iteration_spectral_radius(W, iters=iters)
    if sr > 0:
        W = (rho_target / sr) * W
    return W

# ------- ESN tanítás -------
def train_esn(u, res_size=300, in_scale=0.5, rho=0.9, reg=1e-4, washout=100, seed=42):
    T = len(u)
    rng = np.random.default_rng(seed)

    Win = (rng.random((res_size, 1)) - 0.5) * 2 * in_scale
    # ritkásabb, stabilabb rezervoár:
    W = rng.random((res_size, res_size)) - 0.5
    mask = rng.random((res_size, res_size)) < 0.1  # 10% sűrűség
    W = W * mask
    W = scale_spectral_radius(W, rho_target=rho, iters=20)

    x = np.zeros(res_size)
    X = []
    Y = []

    # tanító szakasz: teacher-forcing jel (u[t]) → állapot; cél: u[t+1]
    for t in range(T - 1):
        x = np.tanh(Win @ np.array([u[t]]) + W @ x)
        if t > washout:
            X.append(x.copy())
            Y.append(u[t+1])

    X = np.stack(X, axis=0)           # [T-washout-1, res_size]
    Y = np.array(Y)                   # [T-washout-1]
    XT = X.T
    Wout = np.linalg.solve(XT @ X + reg*np.eye(res_size), XT @ Y)

    # visszaadjuk az utolsó tanítási állapotot is (jó kezdés szabadfutáshoz)
    return Win, W, Wout, x.copy()

# ------- ESN előrejelzés (szabadfutás) -------
def predict_esn(u0, Win, W, Wout, steps=200, state0=None):
    state = np.zeros(W.shape[0]) if state0 is None else state0.copy()
    preds = []
    u = u0
    for _ in range(steps):
        state = np.tanh(Win @ np.array([u]) + W @ state)
        y = Wout @ state
        preds.append(y)
        u = y  # generative mode
    return np.array(preds), state

# ------- 3D latens-tér (PCA 3 komponens) -------
def pca_3d_trajectory(states_matrix):
    # states_matrix: [T_state, res_size]
    X = states_matrix - states_matrix.mean(axis=0, keepdims=True)
    # SVD alapú PCA
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:3].T  # első 3 PC
    return Z  # shape [T_state, 3]

# -------------- Streamlit App --------------
def run():
    st.set_page_config(layout="wide")
    st.title("🔮 Echo State Network – Idősor-modellezés és 3D latens tér")

    st.markdown(
        "A **reservoir computing** lényege: a nemlineáris, magas dimenziós **rezervoár** fix, csak a "
        "**kimeneti lineáris olvasót** tanítjuk (ridge-regresszió)."
    )

    # Paraméterek
    st.sidebar.header("⚙️ Paraméterek")
    signal_kind = st.sidebar.selectbox("Bemeneti jel", ["Szinusz", "Mackey–Glass (szintetikus)"])
    T = st.sidebar.slider("Tanulási hossz (T)", 400, 6000, 2000, 100)
    res_size = st.sidebar.slider("Rezervoár méret", 50, 1200, 300, 50)
    in_scale = st.sidebar.slider("Bemeneti skála", 0.05, 2.0, 0.5, 0.05)
    rho = st.sidebar.slider("Spektrálsugár céltart.", 0.1, 1.2, 0.9, 0.05)
    reg = st.sidebar.select_slider("Ridge regulár.", options=[1e-6, 1e-5, 1e-4, 1e-3], value=1e-4)
    washout = st.sidebar.slider("Washout", 50, 2000, 200, 50)
    pred_horizon = st.sidebar.slider("Szabadfutás hossza", 50, 3000, 400, 50)
    val_frac = st.sidebar.slider("Validációs arány", 0.0, 0.5, 0.1, 0.05)
    show_3d = st.sidebar.checkbox("3D latens-trajectória (PCA)", value=True)

    # Jel + split (train/val)
    u = gen_input_signal(signal_kind, T)
    split = int((1.0 - val_frac) * len(u))
    u_train = u[:split]
    u_val = u[split:] if val_frac > 0 else None

    # Tanító jel
    st.subheader("📈 Tanító jel")
    fig0, ax0 = plt.subplots()
    ax0.plot(u, lw=1.4, label="u(t)")
    ax0.axvspan(0, split, color="tab:green", alpha=0.1, label="Train")
    if val_frac > 0:
        ax0.axvspan(split, len(u), color="tab:orange", alpha=0.1, label="Validation")
    ax0.set_xlabel("t")
    ax0.set_ylabel("u(t)")
    ax0.legend()
    st.pyplot(fig0)

    # Tanítás + állapotrögzítés a 3D-hez
    with st.spinner("Tanítás folyamatban…"):
        Win, W, Wout, last_state = train_esn(
            u_train, res_size=res_size, in_scale=in_scale, rho=rho, reg=reg, washout=washout
        )

        # Állapotsor rögzítése tanítás közben a 3D-hez (teacher forcing)
        x = np.zeros(W.shape[0])
        states = []
        for t in range(len(u_train) - 1):
            x = np.tanh(Win @ np.array([u_train[t]]) + W @ x)
            if t > washout:
                states.append(x.copy())
        states = np.array(states) if len(states) > 0 else None

    st.success("Kész: Wout betanítva.")

    # Szabadfutás előrejelzés a tanító utolsó pontjától
    st.subheader("🔭 Szabadfutás előrejelzés")
    u0 = u_train[-1]
    y_pred, end_state = predict_esn(u0, Win, W, Wout, steps=pred_horizon, state0=last_state)

    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(len(u)), u, label="Jel (u)", lw=1.2)
    ax1.plot(np.arange(len(u_train)-1, len(u_train)-1+pred_horizon), y_pred, label="ESN előrejelzés", lw=2)
    ax1.axvline(len(u_train)-1, color="gray", ls="--", alpha=0.6)
    ax1.legend()
    ax1.set_xlabel("t")
    st.pyplot(fig1)

    # Ha van validációs szakasz, mért hiba (RMSE) a közös szegmensre
    if u_val is not None and len(u_val) > 0:
        comp_len = min(len(u_val), pred_horizon)
        rmse = np.sqrt(np.mean((y_pred[:comp_len] - u_val[:comp_len])**2))
        st.info(f"Validációs RMSE (első {comp_len} lépés): **{rmse:.4f}**")

    # 3D latens tér – PCA
    if show_3d and states is not None and states.shape[0] >= 5:
        st.subheader("🌌 3D latens trajektória (PCA első 3 komponens)")
        Z = pca_3d_trajectory(states)  # [T_state, 3]
        t_idx = np.arange(Z.shape[0])

        fig3d = go.Figure(data=[go.Scatter3d(
            x=Z[:, 0], y=Z[:, 1], z=Z[:, 2],
            mode='markers+lines',
            marker=dict(size=3, color=t_idx, colorscale='Viridis', showscale=True, colorbar=dict(title="Idő")),
            line=dict(width=2)
        )])
        fig3d.update_layout(
            scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
            margin=dict(l=0, r=0, b=0, t=30),
            height=560
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # Tudományos háttér
    st.markdown("### 📚 Tudományos háttér")
    st.latex(r"\mathbf{x}(t+1)=\tanh\!\big(W\mathbf{x}(t)+W_{in}u(t)\big),\qquad \hat{y}(t)=W_{out}\,\mathbf{x}(t)")
    st.latex(r"W_{out}=(X^\top X+\lambda I)^{-1}X^\top Y")
    st.markdown(
        "- A **rezervoár** nemlineáris, magas dimenziós beágyazást ad; az **echo state property** miatt "
        "megfelelő spektrálsugár (**ρ≲1**) mellett a dinamika feledékeny és stabil.\n"
        "- A 3D PCA a rezervoár-állapotok **fő komponenseit** mutatja: ha jól strukturált, a latens térben "
        "simítható, kvázi-alacsony dimenziós **trajektória** alakul ki.\n"
        "- A szabadfutás **mintázatot folytat** (generative mode), nem földi igazságot rekonstruál."
    )

# ReflectAI kompat
app = run
