reflectai_app.py - foindito fajl (emoji- es unicode-mentes)

import streamlit as st from datetime import datetime

Modulok importálása

from modules.kuramoto_sim import run as run_kuramoto from modules.hebbian_learning import run as run_hebbian from modules.xor_prediction import run as run_xor from modules.kuramoto_hebbian_sim import run as run_kuramoto_hebbian from modules.topo_protect import run as run_topo_protect from modules.lorenz_sim import run as run_lorenz_sim from modules.mlp_predict_lorenz import run as run_lorenz_pred from modules.berry_curvature import run as run_berry from modules.noise_robustness import run as run_noise from modules.esn_prediction import run as run_esn from modules.plasticity_dynamics import run as run_plasticity from modules.fractal_dimension import run as run_fractal from modules.memory_landscape import run as run_memory_landscape from modules.graph_sync_analysis import run as run_graph_sync_analysis from modules.persistent_homology import run as run_homology from modules.help_module import run as run_help from modules.data_upload import run as run_data_upload from modules.lyapunov_spectrum import run as run_lyapunov_spectrum from modules.insight_learning import run as run_insight_learning from modules.generative_kuramoto import run as run_generative_kuramoto from modules.reflection_modul import run as run_reflection

Streamlit beállítások

st.set_page_config(page_title="Neurolab AI - Scientific Playground Sandbox", page_icon="🧠", layout="wide") st.title("Neurolab AI - Scientific Playground Sandbox") st.markdown("Valassz egy modult a bal oldali savbol a vizualizacio inditasahoz.") st.text_input("Megfigyeles vagy jegyzet (opcionalis):")

Modulvalaszto

st.sidebar.title("Modulvalaszto") module_name = st.sidebar.radio("Kerlek valassz:", ( "Kuramoto szinkronizacio", "Hebbian tanulas", "XOR predikcio", "Kuramoto-Hebbian halo", "Topologiai szinkronizacio", "Lorenz szimulacio", "Lorenz predikcio", "Topologiai vedettseg (Chern-szam)", "Topologiai Chern-szam analizis", "Zajturess es szinkronizacios robusztussag", "Echo State Network (ESN) predikcio", "Hebbian plaszticitas dinamikaja", "Szinkronfraktal dimenzioanalizis", "Belatas alapu tanulas (Insight Learning)", "Generativ Kuramoto halo", "Memoria tajkep (Pro)", "Grafalapu szinkronanalizis", "Perzisztens homologia", "Lyapunov spektrum", "Adatfeltoltes modul", "Napi onreflexio", "Sugo / Help" ))

Modulok futtatasa feltetel szerint

if module_name == "Kuramoto szinkronizacio": coupling = st.slider("Kapcsolasi erosseg (K)", 0.0, 10.0, 2.0) num_osc = st.number_input("Oszillatorok szama", min_value=2, max_value=100, value=10) run_kuramoto(coupling, num_osc)

elif module_name == "Hebbian tanulas": learning_rate = st.slider("Tanulasi rata", 0.001, 1.0, 0.1) num_neurons = st.number_input("Neuronok szama", min_value=2, max_value=100, value=10) run_hebbian(learning_rate, num_neurons)

elif module_name == "XOR predikcio": hidden_size = st.slider("Rejtett reteg merete", 1, 10, 2) learning_rate = st.slider("Tanulasi rata", 0.001, 1.0, 0.1) epochs = st.number_input("Epochok szama", 100, 10000, 1000, step=100) note = st.text_input("Megjegyzes (opcionalis)") run_xor(hidden_size, learning_rate, epochs, note)

elif module_name == "Kuramoto-Hebbian halo": run_kuramoto_hebbian()

elif module_name == "Topologiai szinkronizacio": run_topo_protect()

elif module_name == "Lorenz szimulacio": run_lorenz_sim()

elif module_name == "Lorenz predikcio": run_lorenz_pred()

elif module_name == "Topologiai vedettseg (Chern-szam)": run_berry()

elif module_name == "Topologiai Chern-szam analizis": run_berry()

elif module_name == "Zajturess es szinkronizacios robusztussag": run_noise()

elif module_name == "Echo State Network (ESN) predikcio": run_esn()

elif module_name == "Hebbian plaszticitas dinamikaja": run_plasticity()

elif module_name == "Szinkronfraktal dimenzioanalizis": run_fractal()

elif module_name == "Belatas alapu tanulas (Insight Learning)": trials = st.slider("Probalkozasok szama", 1, 20, 5) pause_time = st.slider("Megallas idotartama (mp)", 0.0, 5.0, 1.0) complexity = st.selectbox("Feladat komplexitasa", ["alacsony", "kozepes", "magas"]) run_insight_learning(trials, pause_time, complexity)

elif module_name == "Generativ Kuramoto halo": run_generative_kuramoto()

elif module_name == "Memoria tajkep (Pro)": run_memory_landscape()

elif module_name == "Grafalapu szinkronanalizis": run_graph_sync_analysis()

elif module_name == "Perzisztens homologia": run_homology()

elif module_name == "Lyapunov spektrum": run_lyapunov_spectrum()

elif module_name == "Adatfeltoltes modul": run_data_upload()

elif module_name == "Napi onreflexio": run_reflection()

elif module_name == "Sugo / Help": run_help()

