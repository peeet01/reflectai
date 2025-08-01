import streamlit as st
from datetime import datetime
import importlib
import os  # 📁 Fájlok listázásához

# 💅 Stílus betöltése
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")  # vagy "static/style.css", ha ott van

# 🌐 Metaadat – ez legyen az első Streamlit hívás!
st.set_page_config(page_title="Neurolab AI", layout="wide")

# 📁 Modul-kategóriák
module_categories = {
    "📈 Vizualizációk": {
        "Fractal Dimension": "fractal_dimension",
        "Hebbian Learning Viz": "hebbian_learning_viz",
        "Lyapunov Spectrum": "lyapunov_spectrum",
        "Persistent Homology": "persistent_homology",
        "Memory Landscape": "memory_landscape",
        "Fractal Explorer": "fractal_explorer",
        "Berry Curvature": "berry_curvature",
        "Neural Entropy": "neural_entropy",
        "Graph Sync Analysis": "graph_sync_analysis",
        "Criticality Explorer": "criticality_explorer",
        "Autoencoder Visualization": "autoencoder_vis",
    },
    "🧠 Tanulási algoritmusok": {
        "Hebbian Learning": "hebbian_learning",
        "Insight Learning": "insight_learning",
        "XOR Prediction": "xor_prediction",
        "MLP Predict Lorenz": "mlp_predict_lorenz",
        "Oja Learning": "oja_learning",
        "STDP Learning": "stdp_learning",
        "BCM tanulás": "bcm_learning",
        "Spiking Neural Network": "snn_simulation",
        "ESN Prediction": "esn_prediction",
        "Critical Hebbian": "critical_hebbian",
        "Information Bottleneck": "information_bottleneck",
        
    },
    "⚗️ Szimulációk és dinamikák": {
        "Kuramoto Sim": "kuramoto_sim",
        "Kuramoto Hebbian Sim": "kuramoto_hebbian_sim",
        "Generative Kuramoto": "generative_kuramoto",
        "Lorenz Sim": "lorenz_sim",
        "Plasticity Dynamics": "plasticity_dynamics",
        "Noise Robustness": "noise_robustness",
        "Ising Sim": "ising_sim",
        "Boltzmann Machine": "boltzmann_machine",
        "Laboratory GAN": "lab_gan_module",
         
    },
    "🧪 Adatfeltöltés és predikciók": {
        "Data Upload": "data_upload",
    },
    "📚 Egyéb / Segéd modulok": {
        "Help": "help_module",
        "Questions": "questions",
        "Reflection Modul": "reflection_modul"
    }
}

# ➕ Kezdőlapot hozzáadjuk
main_menu = "🏠 Kezdőlap"

# 🧭 Oldalsáv felépítése – először kategória, aztán modul
st.sidebar.subheader("🧪 Modulválasztó")
category_names = [main_menu] + list(module_categories.keys())
selected_category = st.sidebar.radio("Kategória:", category_names)

# 🏠 Kezdőlap
if selected_category == main_menu:
    with st.container():
        st.image("static/nyitokep.png", use_container_width=True)
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image("static/logo.png", width=180)
        st.markdown("<h3>Neurolab AI – Intelligens szimulációs platform</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("👉 Válassz modult a bal oldali menüből.")
        with st.expander("📘 Mi ez a platform?", expanded=True):
            st.markdown("""
            **Neurolab AI Sandbox** egy interaktív szimulációs platform, amely az **agyműködés**, **tanulás**, **hálózati dinamika** és **komplex rendszerek** megértését segíti elő.  
            A cél, hogy **vizuálisan értelmezhető módon** tanulmányozzuk az **idegtudományi és gépi tanulási modelleket**.

            🧪 **Kinek ajánlott?**  
            - Kutatók  
            - Oktatók  
            - Egyetemi hallgatók  
            - Adattudósok, AI-rajongók

            📊 A modulok különféle területeket fednek le: tanulási algoritmusok, szinkronizációs rendszerek, vizualizációk és adatelemzések.
            """)
else:
    modules = module_categories[selected_category]
    selected_title = st.sidebar.radio("Modul:", list(modules.keys()))
    selected_module_name = modules[selected_title]
    try:
        # 🔁 Modul dinamikus betöltése
        module = importlib.import_module(f"modules.{selected_module_name}")
        if hasattr(module, "app"):
            module.app()
        else:
            st.error(f"❌ A(z) `{selected_module_name}` modul nem tartalmaz `app()` nevű függvényt.")
    except ModuleNotFoundError:
        st.error(f"📦 A(z) `{selected_module_name}` modul nem található a `modules/` mappában.")
    except Exception as e:
        st.error(f"🚨 Hiba történt a(z) `{selected_title}` modul betöltésekor:")
        st.exception(e)
