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
from modules.persistent_homology import run as run_homology
from modules.lyapunov_spectrum import run as run_lyapunov
from modules.memory_landscape import run as run_memory  # ✅ ÚJ MODUL

# App beállítás
st.set_page_config(page_title="ReflecAI – Szinkronizáció és MI", layout="wide")
st.title("🌐 ReflecAI – Szinkronizáció és Mesterséges Intelligencia")
st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció elindításához.")

# Oldalsáv – modulválasztó
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
    "Szinkronfraktál dimenzióanalízis",
    "Perzisztens homológia",
    "Lyapunov-spektrum vizsgálat",
    "Memória tájkép vizualizáció"  # ✅ ÚJ MENÜPONT
))

# Modulok futtatása
if module_name == "Kuramoto szinkronizáció":
    run_kuramoto()

elif module_name == "Hebbian tanulás":
    run_hebbian()

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

elif module_name == "Perzisztens homológia":
    run_homology()

elif module_name == "Lyapunov-spektrum vizsgálat":
    run_lyapunov()

elif module_name == "Memória tájkép vizualizáció":
    run_memory()
