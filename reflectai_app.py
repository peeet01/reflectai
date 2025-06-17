import streamlit as st
import matplotlib.pyplot as plt
from modules.kuramoto_sim import run_kuramoto_simulation
from modules.hebbian_learning import run_hebbian_learning_with_noise
from modules.hebbian_learning_visual import run_hebbian_learning_with_visual
from modules.xor_prediction import run_xor_prediction
from modules.lorenz_sim import generate_lorenz_data
from modules.predict_lorenz import predict_lorenz
from modules.mlp_predict_lorenz import predict_lorenz_mlp
from modules.kuramoto_hebbian_dynamic import run_adaptive_kuramoto_hebbian
from modules.hebbian_topo_kuramoto import run_topo_adaptive_kuramoto

st.set_page_config(page_title="ReflectAI Pro", layout="wide")

st.sidebar.title("🔧 Modulválasztó")
module = st.sidebar.radio("Válassz modult:", [
    "🏠 Főoldal",
    "🌐 Kuramoto szinkronizáció",
    "🧬 Hebbian tanulás",
    "📉 Súlydinamika",
    "🌀 Lorenz predikció (Ridge)",
    "🧠 Lorenz predikció (MLP)",
    "🔁 XOR tanulás",
    "🔄 Adaptív Kuramoto–Hebbian",
    "🌐 Topológikus Kuramoto–Hebbian"
])

if module == "🏠 Főoldal":
    st.title("🧠 ReflectAI Pro")
    st.markdown("**Tudományos MI-szimulációs platform**")
    st.write("Ez az alkalmazás a neuromorf, fotonikus és prediktív rendszerek kutatásához készült.")
    st.success("Használj oldalt modult, állítsd be a paramétereket, és szimulálj!")

elif module == "🌐 Kuramoto szinkronizáció":
    st.header("🌐 Kuramoto szinkronizáció")
    fig1, steps_needed = run_kuramoto_simulation()
    st.pyplot(fig1)
    st.success(f"Szinkronizációs idő: {steps_needed} iteráció")

elif module == "🧬 Hebbian tanulás":
    st.header("🧬 Hebbian tanulás zajmodellel")
    learning_rate = st.slider("Tanulási ráta", 0.01, 1.0, 0.1, 0.01)
    noise_level = st.slider("Zaj szintje", 0.0, 1.0, 0.1, 0.01)
    iterations = st.slider("Iterációk száma", 10, 1000, 100, 10)
    fig2 = run_hebbian_learning_with_noise(learning_rate, noise_level, iterations)
    st.pyplot(fig2)

elif module == "📉 Súlydinamika":
    st.header("📉 Hebbian súlydinamika")
    fig3 = run_hebbian_learning_with_visual(
        learning_rate=0.1,
        noise_level=0.1,
        iterations=100
    )
    st.pyplot(fig3)

elif module == "🌀 Lorenz predikció (Ridge)":
    st.header("🌀 Lorenz-attraktor predikció (Ridge)")
    t, true_states = generate_lorenz_data()
    pred_states = predict_lorenz(true_states, window=5)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(*true_states.T)
    ax.set_title("Valódi Lorenz pálya")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(*pred_states.T, color='orange')
    ax2.set_title("Predikált pálya (Ridge)")
    st.pyplot(fig)

elif module == "🧠 Lorenz predikció (MLP)":
    st.header("🧠 Lorenz-attraktor predikció (MLP)")
    t, true_states = generate_lorenz_data()
    pred_states = predict_lorenz_mlp(true_states, window=5)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(*true_states.T)
    ax.set_title("Valódi Lorenz pálya")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(*pred_states.T, color='green')
    ax2.set_title("Predikált pálya (MLP)")
    st.pyplot(fig)

elif module == "🔁 XOR tanulás":
    st.header("🔁 XOR predikciós feladat")
    accuracy = run_xor_prediction()
    st.success(f"Predikciós pontosság: {accuracy:.2f} %")

elif module == "🔄 Adaptív Kuramoto–Hebbian":
    st.header("🔄 Adaptív Kuramoto–Hebbian háló")
    fig = run_adaptive_kuramoto_hebbian()
    st.pyplot(fig)

elif module == "🌐 Topológikus Kuramoto–Hebbian":
    st.header("🌐 Topológikus Kuramoto–Hebbian háló (Kisvilág)")
    fig = run_topo_adaptive_kuramoto()
    st.pyplot(fig)