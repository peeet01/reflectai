import streamlit as st

CSS betöltése

with open("style.css") as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

Modulok importálása

from modules.kuramoto_sim import run as run_kuramoto from modules.hebbian_learning import run as run_hebbian from modules.xor_prediction import run as run_xor from modules.kuramoto_hebbian_sim import run as run_kuramoto_hebbian from modules.topo_protect import run as run_topo_protect from modules.lorenz_sim import run as run_lorenz_sim from modules.predict_lorenz import run as run_lorenz_pred from modules.berry_curvature import run as run_berry from modules.noise_robustness import run as run_noise from modules.esn_prediction import run as run_esn from modules.plasticity_dynamics import run as run_plasticity from modules.fractal_dimension import run as run_fractal from modules.insight_learning import run as run_insight_learning from modules.generative_kuramoto import run as run_generative_kuramoto from modules.memory_landscape import run as run_memory_landscape from modules.graph_sync_analysis import run as run_graph_sync_analysis

Alkalmazás címe és bevezető

st.set_page_config(page_title="NeuroLab AI – Scientific Playground Sandbox", layout="wide", page_icon="🧠") st.title("\ud83e\udde0 NeuroLab AI – Scientific Playground Sandbox") st.markdown("Fedezd fel a mesterséges intelligencia, szinkronizáció és adaptív rendszerek világát interaktív szimulációkkal, modellezéssel és vizualizációval – minden egy helyen.")

Üres szövegmező a megfigyelésekhez

st.text_input("\ud83d\udcdd Megfigyelés vagy jegyzet (opcionális):")

Modulválasztó

st.sidebar.title("\ud83e\uddea Sandbox Modulválasztó") module_name = st.sidebar.radio("Kérlek válassz:", ( "Kuramoto szinkronizáció", "Hebbian tanulás", "XOR predikció", "Kuramoto–Hebbian háló", "Topológiai szinkronizáció", "Lorenz szimuláció", "Lorenz predikció", "Topológiai védettség (Chern-szám)", "Topológiai Chern–szám analízis", "Zajtűrés és szinkronizációs robusztusság", "Echo State Network (ESN) predikció", "Hebbian plaszticitás dinamikája", "Szinkronfraktál dimenzióanalízis", "Belátás alapú tanulás (Insight Learning)", "Generatív Kuramoto hálózat", "Memória tájkép (Pro)", "Gráf szinkronizációs analízis" ))

Modulok futtatása

if module_name == "Kuramoto szinkronizáció": st.subheader("\ud83e\udd1d Kuramoto paraméterek") coupling = st.slider("Kapcsolási erősség (K)", 0.0, 10.0, 2.0) num_osc = st.number_input("Oszcillátorok száma", min_value=2, max_value=100, value=10) run_kuramoto(coupling, num_osc)

elif module_name == "Hebbian tanulás": st.subheader("\ud83e\udde0 Hebbian paraméterek") learning_rate = st.slider("Tanulási ráta", 0.001, 1.0, 0.1) num_neurons = st.number_input("Neuronok száma", min_value=2, max_value=100, value=10) run_hebbian(learning_rate, num_neurons)

elif module_name == "XOR predikció": st.subheader("\ud83e\udde0 XOR tanítása neurális hálóval") hidden_size = st.slider("Rejtett réteg neuronjainak száma", 1, 10, 2) learning_rate = st.slider("Tanulási ráta", 0.001, 1.0, 0.1) epochs = st.number_input("Epochok száma", min_value=100, max_value=10000, value=1000, step=100) note = st.text_input("Megjegyzés (opcionális)") run_xor(hidden_size, learning_rate, epochs, note)

elif module_name == "Kuramoto–Hebbian háló": run_kuramoto_hebbian()

elif module_name == "Topológiai szinkronizáció": run_topo_protect()

elif module_name == "Lorenz szimuláció": run_lorenz_sim()

elif module_name == "Lorenz predikció": run_lorenz_pred()

elif module_name == "Topológiai védettség (Chern-szám)": run_berry()

elif module_name == "Topológiai Chern–szám analízis": run_berry()

elif module_name == "Zajtűrés és szinkronizációs robusztusság": run_noise()

elif module_name == "Echo State Network (ESN) predikció": run_esn()

elif module_name == "Hebbian plaszticitás dinamikája": run_plasticity()

