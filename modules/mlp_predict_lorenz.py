import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

def create_dataset(data, window=10, target_col=0):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window].flatten())
        y.append(data[i+window][target_col])
    return np.array(X), np.array(y)

def run():
    st.set_page_config(layout="wide")
    st.title("🌪️ Lorenz-rendszer MLP Predikció")

    st.markdown("""
    A **Lorenz-rendszer** a determinisztikus káosz egyik legismertebb példája.  
    Ez a modul bemutatja, hogyan képes egy **többrétegű perceptron (MLP)** a rendszer múltbeli állapotaiból előre jelezni egy jövőbeli értéket.
    """)

    # Paraméterek
    st.sidebar.header("🎚️ Paraméterek")
    steps = st.sidebar.slider("Szimulációs lépések", 500, 5000, 1000, step=100)
    dt = st.sidebar.number_input("Időlépés (dt)", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
    window = st.sidebar.slider("Ablakméret", 5, 50, 10)
    hidden_layer_size = st.sidebar.slider("Rejtett réteg méret", 5, 200, 50)
    max_iter = st.sidebar.slider("Maximális iterációk", 100, 5000, 1000, step=100)
    test_ratio = st.sidebar.slider("Tesztelési arány", 0.1, 0.5, 0.2)
    seed = st.sidebar.number_input("Seed", 0, 9999, 42)
    component = st.sidebar.selectbox("Célkomponens", {"x": 0, "y": 1, "z": 2}, format_func=lambda x: ["x", "y", "z"][x])

    # Adatok
    data, t_eval = generate_lorenz_data(n_steps=steps, dt=dt)
    X, y_target = create_dataset(data, window=window, target_col=component)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_target[:split], y_target[split:]

    # Modell
    model = MLPRegressor(hidden_layer_sizes=(hidden_layer_size,), max_iter=max_iter, random_state=seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Vizualizáció: előrejelzés
    st.subheader("📈 Előrejelzés vizualizáció")
    fig, ax = plt.subplots()
    ax.plot(y_test, label="Valódi", linewidth=2)
    ax.plot(predictions, label="Predikció", linestyle="dashed")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel(f"{['x', 'y', 'z'][component]} komponens")
    ax.legend()
    st.pyplot(fig)

    # Lorenz attraktor
    st.subheader("🌐 Lorenz attraktor (valódi adatok)")
    fig3d = go.Figure(data=[go.Scatter3d(x=data[:,0], y=data[:,1], z=data[:,2],
                                         mode='lines', line=dict(color='blue', width=2))])
    fig3d.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    st.plotly_chart(fig3d, use_container_width=True)

    # Teljesítmény
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    st.subheader("🎯 Teljesítménymutatók")
    st.markdown(f"""
    - **R² (pontosság):** {r2:.3f}  
    - **Átlagos négyzetes hiba (MSE):** {mse:.5f}  
    - **Átlagos abszolút hiba (MAE):** {mae:.5f}  
    - **Rejtett réteg méret:** {hidden_layer_size}  
    """)

    # Loss görbe, ha van
    if hasattr(model, "loss_curve_"):
        st.subheader("📉 Tanulási görbe (loss)")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(model.loss_curve_, color="crimson")
        ax_loss.set_xlabel("Iteráció")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss alakulása")
        st.pyplot(fig_loss)

    # Súlymátrix
    st.subheader("🧮 Súlymátrix vizualizáció")
    flat_weights = np.concatenate([w.flatten() for w in model.coefs_])
    fig_heat, ax_heat = plt.subplots(figsize=(6,1))
    sns.heatmap(flat_weights.reshape(1, -1), cmap="coolwarm", cbar=True, ax=ax_heat)
    ax_heat.set_title("Súlyok hőtérképe")
    st.pyplot(fig_heat)

    # CSV export
    st.subheader("💾 CSV export")
    df_out = pd.DataFrame({
        "y_valódi": y_test,
        "y_predikció": predictions
    })
    csv = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Letöltés CSV-ben", data=csv, file_name="lorenz_predictions.csv")

    # Tudományos háttér
    st.markdown("### 📘 Tudományos háttér")
    st.latex(r"""
    \begin{aligned}
    \frac{dx}{dt} &= \sigma (y - x) \\
    \frac{dy}{dt} &= x (\rho - z) - y \\
    \frac{dz}{dt} &= xy - \beta z
    \end{aligned}
    """)
    st.latex(r"""
    \hat{x}_{t+1} = f(x_t, x_{t-1}, \dots, x_{t-w})
    """)
    st.markdown("""
    A **Lorenz-rendszer** egy kaotikus dinamika, amely hosszú távon nem determinisztikusan viselkedik.  
    Az MLP célja, hogy megtanulja az időbeli mintázatokat a múltbeli állapotok alapján, és becslést adjon a jövőbeli értékre.

    A modell teljesítményét a **R²**, **MSE** és **MAE** mutatók jellemzik, a tanulás minőségét pedig a veszteséggörbe és a súlyok eloszlása.
    """)

# ReflectAI kompatibilitás
app = run
