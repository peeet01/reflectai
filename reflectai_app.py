import streamlit as st
import yaml
from yaml.loader import SafeLoader
from utils.metadata_loader import load_metadata
from modules.journal import journal_module
from modules.reflection_template import reflection_template_module

# Konfiguráció betöltése
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

st.set_page_config(page_title="ReflectAI – Scientific Reflection", layout="wide")
st.title("🧠 ReflectAI – Scientific Reflection")

# Navigációs menü
st.sidebar.title("Navigáció")
page = st.sidebar.selectbox("Válassz modult:", ["Kutatási napló", "Reflexió sablon"])

MODULES = {
    "Kutatási napló": journal_module,
    "Reflexió sablon": reflection_template_module,
}

# Metaadatok betöltése és megjelenítése
metadata = load_metadata(page)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Verzió:** {metadata.get('version', 'N/A')}")
st.sidebar.markdown(f"**Fejlesztő:** {metadata.get('author', 'Ismeretlen')}")

# Modul futtatása
if page in MODULES:
    MODULES[page]()
else:
    st.error("❌ Modul nem található.")
