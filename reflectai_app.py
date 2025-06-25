import streamlit as st
from modules.modules_registry import MODULES

# Oldal beállítás
st.set_page_config(page_title="Neurolab AI – Scientific Reflection", layout="wide")
st.title("🧠 Neurolab AI – Scientific Reflection")
st.markdown("Válassz egy modult a bal oldali menüből.")

# Oldalsáv – Modulválasztó
st.sidebar.title("📂 Modulválasztó")
module_key = st.sidebar.selectbox("Válaszd ki a betölteni kívánt modult:", list(MODULES.keys()))

# Modul betöltése
if module_key in MODULES:
    MODULES[module_key]()
else:
    st.error("❌ A kiválasztott modul nem található.")
