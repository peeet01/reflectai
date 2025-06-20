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
from modules.insight_learning import run as run_insight_learning
from modules.generative_kuramoto import run as run_generative_kuramoto
from modules.generative_x import run as run_generative_x

# Oldal beállítása
st.set_page_config(page_title="ReflecAI - Szinkronizáció és MI", layout="wide")
st.title("🌐 ReflecAI - Szinkronizáció és Mesterséges Intelligencia")
st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció indításához.")

# 💬 Szövegdoboz megfigyelésekhez (opcionális)
user_input = st.text_input("💬 Megjegyzés vagy kérdés (nem nyelvi feldolgozás):")
if user_input:
    st.info(f"📌 Megjegyzés rögzítve: {user_input}")

# Modulválasztó menü
st.sidebar.title("📂 Modulválasztó")
module_name = st.sidebar.radio("Válassz egy modult:", (
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
    "Szinkronfraktál dimenzióanalízis",
    "Belátás alapú tanulás (Insight Learning)",
    "Generatív Kuramoto",
    "Generatív X-szimuláció"
))

# Modulok futtatása
if module_name == "Kuramoto szinkronizáció":
    st.subheader("🧭 Kuramoto paraméterek")
    coupling = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    num_osc = st.number_input("Oszcillátorok száma", min_value=2, max_value=100, value=10)
    run_kuramoto(coupling, num_osc)

elif module_name == "Hebbian tanulás":
    st.subheader("🧠 Hebbian paraméterek")
    learning_rate = st.slider("Tanulási ráta", 0.001, 1.0, 0.1)
    num_neurons = st.number_input("Neuronok száma", min_value=2, max_value=100, value=10)
    run_hebbian(learning_rate, num_neurons)

elif module_name == "XOR predikció":
    st.subheader("🧠 XOR tanítása neurális hálóval")
    hidden_size = st.slider("Rejtett réteg neuronjainak száma", 1, 10, 2)
    learning_rate = st.slider("Tanulási ráta", 0.001, 1.0, 0.1)
    epochs = st.number_input("Epochok száma", min_value=100, max_value=10000, value=1000, step=100)
    note = st.text_input("Megjegyzés (opcionális)")
    run_xor(hidden_size, learning_rate, epochs, note)

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

elif module_name == "Belátás alapú tanulás (Insight Learning)":
    st.subheader("💡 Belátás alapú tanulási szimuláció")
    trials = st.slider("Próbálkozások száma", 1, 20, 5)
    pause_time = st.slider("Megállás hossza (másodperc)", 0.0, 5.0, 1.0)
    complexity = st.selectbox("Feladat komplexitása", ["alacsony", "közepes", "magas"])
    run_insight_learning(trials, pause_time, complexity)

elif module_name == "Generatív Kuramoto":
    st.subheader("🌱 Generatív Kuramoto szimuláció")
    run_generative_kuramoto()

elif module_name == "Generatív X-szimuláció":
    st.subheader("🔄 Generatív X típusú dinamika")
    run_generative_x()
