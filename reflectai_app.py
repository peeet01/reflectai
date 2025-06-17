import streamlit as st
from modules.kuramoto_sim import run_kuramoto_simulation
from modules.kuramoto_hebbian import run_adaptive_kuramoto
from modules.hebbian_learning import run_hebbian_learning_with_noise
from modules.hebbian_learning_visual import plot_hebbian_learning
from modules.lorenz_sim import generate_lorenz_data
from modules.predict_lorenz import predict_lorenz
from modules.mlp_predict_lorenz import run_lorenz_mlp
from modules.xor_prediction import run_xor_prediction

# Beállítások
st.set_page_config(page_title="ReflectAI Pro", page_icon="🧠")
st.title("🧠 ReflectAI Pro – Fejlett MI szimulációk")

user_input = st.text_input("💬 Add meg kutatási kérdésed vagy utasításod:")

if user_input:
    st.subheader("🔎 Elemzés")
    st.write("A rendszer a megadott kérdés alapján különböző MI-komponenseket aktivál.")

# --- Kuramoto szinkronizáció ---
st.header("🌐 Klasszikus Kuramoto szinkronizáció")
fig1, steps = run_kuramoto_simulation()
st.pyplot(fig1)
st.success(f"Szinkronizációs iterációk száma: {steps}")

# --- Adaptív Hebbian Kuramoto háló ---
st.header("🧠 Adaptív Hebbian–Kuramoto hálózat")
fig2, metrics = run_adaptive_kuramoto()
st.pyplot(fig2)
st.info(f"Koherencia: {metrics['coherence']:.2f} | Iterációk: {metrics['steps']}")

# --- Hebbian tanulás (vizualizációval) ---
st.header("🔬 Hebbian tanulás zajmodellel")
fig3 = run_hebbian_learning_with_noise()
st.pyplot(fig3)

st.header("🎯 Hebbian tanulási vizualizáció")
fig4 = plot_hebbian_learning()
st.pyplot(fig4)

# --- Lorenz attraktor predikció (klasszikus és MLP) ---
st.header("🌪️ Lorenz attraktor predikció")
fig5 = predict_lorenz()
st.pyplot(fig5)

st.header("🧠 MLP predikció Lorenz adatokra")
accuracy = run_lorenz_mlp()
st.success(f"MLP predikciós pontosság: {accuracy:.2f} %")

# --- XOR feladat ---
st.header("🧩 XOR tanulási feladat")
acc_xor = run_xor_prediction()
st.success(f"XOR predikciós pontosság: {acc_xor:.2f} %")
