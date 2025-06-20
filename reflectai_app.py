import streamlit as st

# CSS stílus beillesztése (pl. oldalikonhoz, színekhez, térközhöz)
st.markdown("""
    <style>
        .main > div {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        section[data-testid="stSidebar"] h3 {
            font-size: 18px !important;
            color: black !important;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1 {
            font-size: 30px !important;
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

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
from modules.memory_landscape import run as run_memory_landscape
from modules.graph_sync_analysis import run as run_graph_sync_analysis

# Alkalmazás beállításai
st.set_page_config(page_title="ReflecAI - Szinkronizáció és MI", layout="wide")

# Főcím agy ikonnal
st.markdown("<h1 style='font-size:30px; color:black;'>🧠 ReflecAI - Szinkronizáció és Mesterséges Intelligencia</h1>", unsafe_allow_html=True)

# Elválasztó
st.markdown("---", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# Megfigyelési szövegmező
st.text_input("📝 Megfigyelés vagy jegyzet (opcionális):")

# Oldalsáv – Modulválasztó
st.sidebar.markdown("<h3 style='font-size:18px; color:black;'>📊 Modulválasztó</h3>", unsafe_allow_html=True)
module_name = st.sidebar.radio("Válassz modult:", (
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
    "Gráf szinkronizációs analízis"
))

# Térköz
st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

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

elif module_name == "Generatív Kuramoto hálózat":
    run_generative_kuramoto()

elif module_name == "Memória tájkép (Pro)":
    run_memory_landscape()

elif module_name == "Gráf szinkronizációs analízis":
    run_graph_sync_analysis()
