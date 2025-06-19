import streamlit as st

# Modulok importálása
from modules.kuramoto_sim import run as run_kuramoto
from modules.hebbian_learning import run as run_hebbian
from modules.xor_prediction import run as run_xor
from modules.kuramoto_hebbian_sim import run as run_kuramoto_hebbian
from modules.topo_protect import run as run_topo_protect
from modules.lorenz_sim import run as run_lorenz_sim
from modules.predict_lorenz import run as run_lorenz_pred
from modules.berry_curvature import run as run_berry
from modules.noise_robustness import run as run_noise
from modules.esn_prediction import run as run_esn
from modules.plasticity_dynamics import run as run_plasticity
from modules.fractal_dimension import run as run_fractal

# Oldal beállítása
st.set_page_config(page_title="ReflecAI - Szinkronizáció és MI", layout="wide")
st.title("🌐 ReflecAI - Szinkronizáció és Mesterséges Intelligencia")
st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció indításához.")

# Modulválasztó
st.sidebar.title("📂 Modulválasztó")
module_name = st.sidebar.radio("Kérlek válassz:", (
    "Kuramoto szinkronizáció",
    "Hebbian tanulás",
    "XOR predikció",
    "Kuramoto–Hebbian háló",
    "Topológiai szinkronizáció",
    "Lorenz szimuláció",
    "Lorenz predikció",
    "Topológiai védettség (Chern-szám)",
    "Topológiai Chern–szám analízis",
    "Zajtűrés és szinkronizációs robusztusság",
    "Echo State Network (ESN) predikció",
    "Hebbian plaszticitás dinamikája",
    "Szinkronfraktál dimenzióanalízis"
))

# Modul futtatás logika
if module_name == "Kuramoto szinkronizáció":
    st.subheader("🌀 Kuramoto szinkronizáció - Paraméterek")
    num_oscillators = st.slider("Oszcillátorok száma", 2, 100, 10)
    coupling_strength = st.slider("Kapcsolódási erősség", 0.0, 5.0, 1.0)
    time_steps = st.number_input("Szimuláció időlépései", min_value=10, max_value=10000, value=200)
    user_comment = st.text_input("Megjegyzés (opcionális)")
    run_kuramoto(num_oscillators, coupling_strength, time_steps, user_comment)

elif module_name == "Hebbian tanulás":
    st.subheader("🧠 Hebbian tanulás - Paraméterek")
    num_neurons = st.slider("Neuronok száma", 2, 100, 10)
    learning_rate = st.slider("Tanulási ráta", 0.001, 1.0, 0.1)
    iterations = st.number_input("Iterációk száma", min_value=10, max_value=5000, value=100)
    user_note = st.text_input("Megjegyzés (opcionális)")
    run_hebbian(num_neurons, learning_rate, iterations, user_note)

elif module_name == "XOR predikció":
    run_xor()

elif module_name == "Kuramoto–Hebbian háló":
    run_kuramoto_hebbian()

elif module_name == "Topológiai szinkronizáció":
    run_topo_protect()

elif module_name == "Lorenz szimuláció":
    run_lorenz_sim()

elif module_name == "Lorenz predikció":
    run_lorenz_pred()

elif module_name == "Topológiai védettség (Chern-szám)":
    run_berry()

elif module_name == "Topológiai Chern–szám analízis":
    run_berry()

elif module_name == "Zajtűrés és szinkronizációs robusztusság":
    run_noise()

elif module_name == "Echo State Network (ESN) predikció":
    run_esn()

elif module_name == "Hebbian plaszticitás dinamikája":
    run_plasticity()

elif module_name == "Szinkronfraktál dimenzióanalízis":
    run_fractal()
