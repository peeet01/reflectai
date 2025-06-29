import streamlit as st
from datetime import datetime
import importlib
import os  # 📁 Fájlok listázásához

# 🌐 Metaadat – ez legyen az első Streamlit hívás!
st.set_page_config(page_title="Neurolab AI", layout="wide")

# 📂 Debug info – segít ellenőrizni a modulbetöltést
st.sidebar.write("📂 Aktuális working directory:", os.getcwd())
st.sidebar.write("📂 modules abs path:", os.path.abspath("modules"))
try:
    st.sidebar.write("📁 modules tartalma:", os.listdir("modules"))
except Exception as e:
    st.sidebar.error(f"Nem tudtam listázni a 'modules' mappát: {e}")

# 📦 Modul-regiszter (modulnév: fájlnév, kiterjesztés nélkül)
modules = {
    "Berry Curvature": "berry_curvature",
    "Data Upload": "data_upload",
    "ESN Prediction": "esn_prediction",
    "Fractal Dimension": "fractal_dimension",
    "Generative Kuramoto": "generative_kuramoto",
    "Graph Sync Analysis": "graph_sync_analysis",
    "Hebbian Learning": "hebbian_learning",
    "Hebbian Learning Viz": "hebbian_learning_viz",
    "Help": "help_module",
    "Insight Learning": "insight_learning",
    "Kuramoto Hebbian Sim": "kuramoto_hebbian_sim",
    "Kuramoto Sim": "kuramoto_sim",
    "Lorenz Sim": "lorenz_sim",
    "Lyapunov Spectrum": "lyapunov_spectrum",
    "Memory Landscape": "memory_landscape",
    "MLP Predict Lorenz": "mlp_predict_lorenz",
    "Noise Robustness": "noise_robustness",
    "Persistent Homology": "persistent_homology",
    "Plasticity Dynamics": "plasticity_dynamics",
    "Questions": "questions",
    "Reflection Modul": "reflection_modul",
    "XOR Prediction": "xor_prediction"
}

# ➕ Kezdőlapot hozzáadjuk a menühöz
menu_titles = ["🏠 Kezdőlap"] + list(modules.keys())

# 🧭 Modulválasztó az oldalsávban
st.sidebar.subheader("🧪 Modulválasztó")
selected_title = st.sidebar.radio("Válassz modult:", menu_titles)

# 🏠 Kezdőlap tartalom
if selected_title == "🏠 Kezdőlap":
    st.image("static/nyitokep.png", use_container_width=True)
    st.title("Üdvözöl a Neurolab AI!")
    st.markdown("👉 Válassz modult a bal oldali menüből.")
else:
    selected_module_name = modules[selected_title]
    try:
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
        # 🔍 Teszt: modulok tényleges betölthetősége
st.subheader("🧪 Modul tesztelés eredményei:")
for name, file in modules.items():
    try:
        m = importlib.import_module(f"modules.{file}")
        if hasattr(m, "app"):
            st.success(f"✅ {file}.py betöltve és van benne `app()`!")
        else:
            st.warning(f"⚠️ {file}.py betöltve, de nincs `app()`!")
    except Exception as e:
        st.error(f"❌ {file}.py nem betölthető: {e}")
