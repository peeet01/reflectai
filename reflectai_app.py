
import streamlit as st
from datetime import datetime
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
from modules.reflection_modul import run as run_reflection

st.set_page_config(
    page_title="Neurolab AI – Scientific Playground Sandbox",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neurolab AI – Scientific Playground Sandbox")
st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció indításához.")
st.text_input("📝 Megfigyelés vagy jegyzet (opcionális):")

st.sidebar.title("📂 Modulválasztó")
module_name = st.sidebar.radio("Kérlek válassz:", (
    "🧠 Napi önreflexió",
    "❓ Súgó / Help"
))

if module_name == "🧠 Napi önreflexió":
    run_reflection()
elif module_name == "❓ Súgó / Help":
    run_help()
