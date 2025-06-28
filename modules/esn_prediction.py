import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def generate_esn_input(data, delay):
    X = []
    y = []
    for i in range(len(data) - delay):
        X.append(data[i:i+delay])
        y.append(data[i+delay])
    return np.array(X), np.array(y)

def esn_predict(X, y, reservoir_size=100, spectral_radius=0.95):
    input_size = X.shape[1]
    Win = np.random.rand(reservoir_size, input_size) - 0.5
    W = np.random.rand(reservoir_size, reservoir_size) - 0.5
    # Spectral radius scaling
    rho = max(abs(np.linalg.eigvals(W)))
    W *= spectral_radius / rho

    states = np.zeros((X.shape[0], reservoir_size))
    x = np.zeros(reservoir_size)
    for t in range(X.shape[0]):
        u = X[t]
        x = np.tanh(np.dot(Win, u) + np.dot(W, x))
        states[t] = x

    Wout = np.dot(np.linalg.pinv(states), y)
    y_pred = np.dot(states, Wout)
    return y_pred

def run():
    st.title("🔁 Echo State Network (ESN) időbeli predikció")
    st.markdown("Az ESN egy visszacsatolt hálózat, amelyet gyakran használnak időfüggő adatok előrejelzésére.")

    st.sidebar.header("Paraméterek")
    series_length = st.sidebar.slider("Idősor hossza", 100, 1000, 300, step=50)
    delay = st.sidebar.slider("Késleltetés (delay)", 2, 50, 10)
    reservoir_size = st.sidebar.slider("Reservoir mérete", 10, 500, 100, step=10)
    spectral_radius = st.sidebar.slider("Spektrális sugár", 0.1, 1.5, 0.95, step=0.05)

    freq = st.sidebar.slider("Szinusz frekvencia", 0.5, 5.0, 1.0, step=0.1)
    noise_level = st.sidebar.slider("Zajszint", 0.0, 0.5, 0.05, step=0.01)

    # Idősor generálás (szinusz + zaj)
    t = np.linspace(0, 10, series_length)
    data = np.sin(2 * np.pi * freq * t) + np.random.normal(0, noise_level, series_length)

    # Bemenet és cél adatok generálása
    X, y = generate_esn_input(data, delay)

    # Predikció
    y_pred = esn_predict(X, y, reservoir_size, spectral_radius)

    # Eredmények kirajzolása
    st.subheader("📈 Predikciós eredmények")
    fig, ax = plt.subplots()
    ax.plot(y, label="Valós")
    ax.plot(y_pred, label="ESN predikció", linestyle='--')
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("Érték")
    ax.legend()
    st.pyplot(fig)

    # Jegyzet hozzáadás
    user_note = st.text_area("📝 Írd le a megfigyeléseid", height=150)
    if user_note:
        st.markdown("### 💬 Megjegyzésed:")
        st.write(user_note)

# 🔧 Dinamikus modulbetöltéshez szükséges
def app():
    run()
