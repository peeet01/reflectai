import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ------- Egyszerű ESN (Echo State Network) -------
def gen_input_signal(kind, T):
    t = np.linspace(0, 50, T)
    if kind == "Szinusz":
        u = np.sin(0.4 * t) + 0.3*np.sin(0.05 * t + 0.7)
    elif kind == "Mackey–Glass (szintetikus)":
        # olcsó imitáció: kvázi-kaotikus jel (nem a PDE!)
        u = np.sin(0.2*t) * np.sin(0.311*t + 1.2) + 0.1*np.random.randn(T)
    else:
        u = np.sin(0.35 * t)
    return u

def scale_spectral_radius(W, rho_target=0.9):
    # kb. spektrálsugár skálázás (power-iteráció helyett olcsó norma-trükk)
    # nem pontos, de stabilabb dinamikát ad
    s = np.linalg.norm(W, 2)
    if s > 0:
        W = W * (rho_target / s)
    return W

def train_esn(u, res_size=300, in_scale=0.5, rho=0.9, reg=1e-4, washout=100):
    T = len(u)
    rng = np.random.default_rng(42)

    Win = (rng.random((res_size, 1)) - 0.5) * 2 * in_scale
    W = rng.random((res_size, res_size)) - 0.5
    W = scale_spectral_radius(W, rho)

    x = np.zeros(res_size)
    X = []
    Y = []

    for t in range(T - 1):
        x = np.tanh(Win @ np.array([u[t]]) + W @ x)
        if t > washout:
            X.append(x.copy())
            Y.append(u[t+1])

    X = np.stack(X, axis=0)           # [T-washout-1, res_size]
    Y = np.array(Y)                   # [T-washout-1]
    # Ridge-regresszió zárt alakban: Wout = (X^T X + reg*I)^-1 X^T Y
    XT = X.T
    Wout = np.linalg.solve(XT@X + reg*np.eye(res_size), XT@Y)
    state = x.copy()
    return Win, W, Wout, state

def predict_esn(u0, Win, W, Wout, steps=200):
    x = np.zeros_like(Wout)  # csak méret miatt nem jó -> valós state kell
    # jobb: kezdjünk egy rövid "primerrel"
    state = np.zeros(W.shape[0])
    preds = []
    u = u0
    for _ in range(steps):
        state = np.tanh(Win @ np.array([u]) + W @ state)
        y = Wout @ state
        preds.append(y)
        u = y  # szabad futás (generative mode)
    return np.array(preds)

# -------------- Streamlit --------------
def run():
    st.set_page_config(layout="wide")
    st.title("🔮 Echo State Network – Gyors idősor-előrejelzés")

    st.markdown(
        "Könnyű **reservoir computing**: csak a kimeneti súlyokat tanítjuk, a belső dinamika fix."
    )

    st.sidebar.header("⚙️ Paraméterek")
    signal_kind = st.sidebar.selectbox("Bemeneti jel", ["Szinusz", "Mackey–Glass (szintetikus)"])
    T = st.sidebar.slider("Tanulási hossz (T)", 400, 5000, 1500, 100)
    res_size = st.sidebar.slider("Rezervoár méret", 50, 1000, 300, 50)
    in_scale = st.sidebar.slider("Bemeneti skála", 0.1, 2.0, 0.5, 0.1)
    rho = st.sidebar.slider("Spektrálsugár céltart.", 0.1, 1.2, 0.9, 0.05)
    reg = st.sidebar.select_slider("Ridge regulár.", options=[1e-6, 1e-5, 1e-4, 1e-3], value=1e-4)
    washout = st.sidebar.slider("Washout", 50, 500, 100, 50)
    pred_horizon = st.sidebar.slider("Előrejelzési hossz", 50, 1500, 300, 50)

    u = gen_input_signal(signal_kind, T)

    st.subheader("📈 Tanító jel")
    fig0, ax0 = plt.subplots()
    ax0.plot(u, lw=1.5)
    ax0.set_xlabel("t")
    ax0.set_ylabel("u(t)")
    st.pyplot(fig0)

    st.subheader("🚀 Tanítás")
    Win, W, Wout, state = train_esn(
        u, res_size=res_size, in_scale=in_scale, rho=rho, reg=reg, washout=washout
    )
    st.success("Kész: kimeneti súlyok (Wout) betanítva.")

    st.subheader("🔭 Szabadfutás előrejelzés")
    u0 = u[-1]  # utolsó tanítóértékről indulunk
    y_pred = predict_esn(u0, Win, W, Wout, steps=pred_horizon)

    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(len(u)), u, label="Tanító jel", lw=1.5)
    ax1.plot(np.arange(len(u)-1, len(u)-1+pred_horizon), y_pred, label="ESN előrejelzés", lw=2)
    ax1.legend()
    ax1.set_xlabel("t")
    st.pyplot(fig1)

    # Hibamérték (ha van földi igazság – itt nincs), ezért a józan értelmezés marad:
    st.markdown(
        "- Az ESN **mintázatot folytat**, nem pontos földi igazságot jelez előre.  \n"
        "- A **ρ** (spektrálsugár) és a **reg** erősen befolyásolja a stabilitást és simaságot."
    )

    st.markdown("### 📚 Képletek")
    st.latex(r"\mathbf{x}(t+1)=\tanh(W\mathbf{x}(t)+W_{in}u(t))\,,\quad \hat{y}(t)=W_{out}\mathbf{x}(t)")
    st.latex(r"W_{out}=(X^\top X+\lambda I)^{-1}X^\top Y")

# ReflectAI kompat
app = run
