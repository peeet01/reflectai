import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.integrate import solve_ivp
import plotly.graph_objects as go


def generate_lorenz_data(n_steps=1000, dt=0.01):
    def lorenz(t, state, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t_span = (0, n_steps * dt)
    y0 = [1.0, 1.0, 1.0]
    t_eval = np.linspace(*t_span, n_steps)
    sol = solve_ivp(lorenz, t_span, y0, t_eval=t_eval)
    return sol.y.T, t_eval


def create_dataset(data, window=10, component=0):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window].flatten())
        y.append(data[i + window][component])
    return np.array(X), np.array(y)


def run():
    st.set_page_config(layout="wide")
    st.title("🌪️ Lorenz-rendszer MLP predikció")

    st.markdown("""
    A **Lorenz-rendszer** a determinisztikus káosz egyik legismertebb példája. Ez a modul bemutatja,
    hogyan képes egy **többrétegű perceptron (MLP)** előrejelezni a rendszer viselkedését kizárólag a múltbeli állapotok alapján.
    """)

    # 🎛️ Paraméterek
    st.sidebar.header("🎚️ Szimulációs és tanulási paraméterek")
    steps = st.sidebar.slider("Szimulációs lépések", 500, 5000, 1000, 100)
    dt = st.sidebar.number_input("Időlépés (dt)", 0.001, 0.1, 0.01, step=0.001)
    window = st.sidebar.slider("Ablakméret (window size)", 5, 50, 10)
    hidden_layer_size = st.sidebar.slider("Rejtett réteg méret", 5, 200, 50)
    max_iter = st.sidebar.slider("Maximális iterációk", 100, 2000, 500, step=100)
    test_ratio = st.sidebar.slider("Tesztelési arány", 0.1, 0.5, 0.2)
    seed = st.sidebar.number_input("Seed", 0, 9999, 42, step=1)
    component_label = st.sidebar.selectbox("Célkomponens", ["x", "y", "z"])
    component = {"x": 0, "y": 1, "z": 2}[component_label]

    # 🔁 Adatok generálása és előkészítés
    data, t_eval = generate_lorenz_data(n_steps=steps, dt=dt)
    X, y_data = create_dataset(data, window=window, component=component)

    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]

    # 🧠 Modell tanítás
    model = MLPRegressor(hidden_layer_sizes=(hidden_layer_size,), max_iter=max_iter, random_state=seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 📈 Előrejelzés
    st.subheader("📊 Valódi vs. prediktált értékek")
    fig, ax = plt.subplots()
    ax.plot(y_test, label="Valódi", linewidth=2)
    ax.plot(predictions, label="MLP predikció", linestyle='dashed')
    ax.set_xlabel("Időlépések")
    ax.set_ylabel(f"{component_label} komponens")
    ax.legend()
    st.pyplot(fig)

    # 🌐 3D Lorenz vizualizáció (valódi)
    st.subheader("🌐 Lorenz attraktor (valódi adatok)")
    x, y_l, z = data.T
    fig3d = go.Figure(data=[go.Scatter3d(x=x, y=y_l, z=z, mode='lines',
                                         line=dict(color='blue', width=2))])
    fig3d.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig3d, use_container_width=True)

    # 📈 Metríkák
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    st.markdown(f"""
    ### 🎯 Modell teljesítménye
    - R² pontosság: **{r2:.3f}**
    - Átlagos négyzetes hiba (MSE): **{mse:.4f}**
    - Rejtett réteg mérete: **{hidden_layer_size}**
    - Cél komponens: **{component_label}**
    """)

    # 📁 CSV export
    st.subheader("💾 Előrejelzések letöltése")
    df_out = pd.DataFrame({"y_valódi": y_test, "y_predikció": predictions})
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Letöltés CSV formátumban", data=csv, file_name="lorenz_predictions.csv")

    st.markdown(r"""
    ### 📘 Tudományos háttér

    **Lorenz-egyenletek** (determinista, de kaotikus dinamikával):
    $$
    \begin{aligned}
    \frac{dx}{dt} &= \sigma \,(y - x),\\
    \frac{dy}{dt} &= x(\rho - z) - y,\\
    \frac{dz}{dt} &= xy - \beta z.
    \end{aligned}
    $$

    A rendszer rövid távon jól előrejelezhető, de a pozitív Lyapunov-exponensek miatt **hosszú távon érzékeny a kezdeti feltételekre** (káosz), ezért a hibák exponenciálisan felerősödnek.

    ---

    **Késleltetett beágyazás (Takens) az MLP bemenetéhez**  
    (csúszó ablak a múltbeli mintákból):
    $$
    \mathbf{x}_t \;=\; \big[x_t,\; x_{t-1},\; \dots,\; x_{t-w+1}\big]^\top,
    $$
    ahol \(w\) az ablakméret (window). Általánosabban késleltetéssel \(\tau\):
    $$
    \mathbf{x}_t \;=\; \big[x_t,\; x_{t-\tau},\; \dots,\; x_{t-(w-1)\tau}\big]^\top.
    $$

    **MLP-alapú regresszió (egy-lépéses előrejelzés):**
    $$
    \hat{x}_{t+1} \;=\; f_\theta(\mathbf{x}_t),
    $$
    illetve \(h\)-lépéses horizontnál:
    $$
    \hat{x}_{t+h} \;=\; f_\theta(\mathbf{x}_t).
    $$

    ---

    **Ridge-regressziós baseline** (összehasonlításként):
    $$
    \min_{\mathbf{w}} \;\; \|X\mathbf{w} - \mathbf{y}\|_2^2 \;+\; \alpha \|\mathbf{w}\|_2^2,
    $$
    ahol \(\alpha>0\) a \(L_2\)-regularizáció súlya.

    ---

    **Kiértékelési metrikák:**

    - **RMSE** (gyök-négyzetes átlagos hiba):
    $$
    \mathrm{RMSE} \;=\; \sqrt{\frac{1}{N}\sum_{i=1}^N \big(\hat{x}_i - x_i\big)^2},
    $$

    - **Determinációs együttható**:
    $$
    R^2 \;=\; 1 \;-\; \frac{\sum_{i=1}^N (x_i - \hat{x}_i)^2}{\sum_{i=1}^N (x_i - \bar{x})^2}.
    $$

    > **Megjegyzés:** Kaotikus dinamikán a rövid horizontú (\(h\) kicsi) előrejelzés reális cél; hosszú távon az előrejelzés **eloszlás-szintű** (statisztikai) megfelelésére érdemes törekedni.
    """)

# ReflectAI kompatibilis
app = run
