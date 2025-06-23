import streamlit as st
from datetime import datetime

# Meglévő modulok importálása (minden modult tartalmaz!)
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
from modules.memory_landscape import run as run_memory_landscape
from modules.graph_sync_analysis import run as run_graph_sync_analysis
from modules.help_module import run as run_help
from modules.data_upload import run as run_data_upload
from modules.lyapunov_spectrum import run as run_lyapunov_spectrum
from modules.insight_learning import run as run_insight_learning
from modules.generative_kuramoto import run as run_generative_kuramoto
from modules.persistent_homology import run as run_homology
from modules.reflection_modul import run as run_reflection

# Streamlit oldalbeállítás
st.set_page_config(
    page_title="Neurolab AI – Scientific Playground Sandbox",
    layout="wide"
)

# Oldal címe
st.title("Neurolab AI – Scientific Playground Sandbox")
st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció indításához.")
st.text_input("Megfigyelés vagy jegyzet (opcionális):")

# Oldalsáv modulválasztó
st.sidebar.title("Modulválasztó")
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
    "Szinkronfraktál dimenzióanalízis",
    "Belátás alapú tanulás (Insight Learning)",
    "Generatív Kuramoto hálózat",
    "Memória tájkép (Pro)",
    "Gráfalapú szinkronanalízis",
    "Lyapunov spektrum",
    "Adatfeltöltés modul",
    "Perzisztens homológia",
    "Napi önreflexió",
    "Súgó / Help"
))

# Modulválasztás és futtatás
if module_name == "Kuramoto szinkronizáció":
    coupling = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0)
    num_osc = st.number_input("Oszcillátorok száma", min_value=2, max_value=100, value=10)
    run_kuramoto(coupling, num_osc)

elif module_name == "Hebbian tanulás":
    learning_rate = st.slider("Tanulási ráta", 0.001, 1.0, 0.1)
    num_neurons = st.number_input("Neuronok száma", min_value=2, max_value=100, value=10)
    run_hebbian(learning_rate, num_neurons)

elif module_name == "XOR predikció":
    hidden_size = st.slider("Rejtett réteg mérete", 1, 10, 2)
    learning_rate = st.slider("Tanulási ráta", 0.001, 1.0, 0.1)
    epochs = st.number_input("Epochok száma", 100, 10000, 1000, step=100)
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
    trials = st.slider("Próbálkozások száma", 1, 20, 5)
    pause_time = st.slider("Megállás időtartama (mp)", 0.0, 5.0, 1.0)
    complexity = st.selectbox("Feladat komplexitása", ["alacsony", "közepes", "magas"])
    run_insight_learning(trials, pause_time, complexity)

elif module_name == "Generatív Kuramoto hálózat":
    run_generative_kuramoto()

elif module_name == "Memória tájkép (Pro)":
    run_memory_landscape()

elif module_name == "Gráfalapú szinkronanalízis":
    run_graph_sync_analysis()

elif module_name == "Lyapunov spektrum":
    run_lyapunov_spectrum()

elif module_name == "Adatfeltöltés modul":
    run_data_upload()

elif module_name == "Perzisztens homológia":
    run_homology()

elif module_name == "Napi önreflexió":
    run_reflection()

elif module_name == "Súgó / Help":
    run_help()
