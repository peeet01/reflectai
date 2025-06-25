import streamlit as st
import yaml
from yaml.loader import SafeLoader

# Modulok betöltése
from modules.modules_registry import MODULES

# Oldalcím és beállítás
st.set_page_config(page_title="ReflectAI", layout="wide")
st.title("🧠 ReflectAI - Moduláris kutatóplatform")

# Modulválasztó
module_names = list(MODULES.keys())
selected_module = st.sidebar.selectbox("Modul kiválasztása", module_names)

# Modul futtatása
if selected_module in MODULES:
    try:
        MODULES[selected_module]()  # feltételezve, hogy minden modulnak van egy main() függvénye
    except Exception as e:
        st.error(f"Hiba történt a modul futtatása közben: {e}")
else:
    st.warning("A kiválasztott modul nem található a rendszerben.")
