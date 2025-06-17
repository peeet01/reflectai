import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from modules.kuramoto_sim import run_kuramoto_simulation
from modules.hebbian_learning import run_hebbian_learning_with_noise
from modules.xor_prediction import run_xor_prediction
from modules.lorenz_sim import generate_lorenz_data
from modules.predict_lorenz import predict_lorenz

st.set_page_config(page_title="ReflectAI – Fejlesztett Kutatási Prototípus", page_icon="🧠")
st.title("🧠 ReflectAI – Tudományos szintű MI szimulációk")

user_input = st.text_input("Kérdésed vagy kutatási parancsod:")
if user_input:
    st.subheader("💡 Szabályalapú válasz")
    st.write("Ez a verzió a kutatási komponensekre fókuszál. Szimulált, tematikus válasz érkezik.")
    st.markdown("### 🔍 Önreflexió:")
    st.write("A rendszer bővített metrikák alapján működik: tanulási zaj, szinkronizációs idő, predikciós pontosság.")

# Kuramoto
st.header("🌐 Kuramoto szinkronizáció")
fig1, steps_needed = run_kuramoto_simulation()
st.pyplot(fig1)
st.success(f"Szinkronizációs idő: {steps_needed} iteráció")

# Hebbian
st.header("🧬 Hebbian tanulás zajmodellel")
learning_rate = st.slider("Tanulási ráta", 0.01, 1.0, 0.1, 0.01)
noise_level = st.slider("Zaj szintje", 0.0, 1.0, 0.1, 0.01)
iterations = st.slider("Iterációk száma", 10, 1000, 100, 10)
fig2 = run_hebbian_learning_with_noise(learning_rate, noise_level, iterations)
st.pyplot(fig2)

# XOR
st.header("🧠 XOR predikciós tanulási feladat")
accuracy = run_xor_prediction()
st.success(f"Predikciós pontosság: {accuracy:.2f} %")

# Lorenz attraktor predikció (Ridge)
st.header("🌀 Lorenz-attraktor predikció (Ridge regresszió)")
if st.button("Szimuláció és predikció futtatása"):
    t, true_states = generate_lorenz_data()
    pred_states = predict_lorenz(true_states, window=5, alpha=1.0)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(*true_states.T, label='Valódi')
    ax.set_title("Valódi Lorenz pálya")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(*pred_states.T, label='Predikált', color='orange')
    ax2.set_title("Predikált Lorenz pálya (Ridge)")

    st.pyplot(fig)