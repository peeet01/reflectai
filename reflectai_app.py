import streamlit as st
from modules.modules_registry import MODULES, safe_run

st.set_page_config(page_title="Neurolab AI – Scientific Playground Sandbox", page_icon="🧠", layout="wide")

st.title("🧠 Neurolab AI – Scientific Playground Sandbox")
st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció indításához.")
st.text_input("Megfigyelés vagy jegyzet (opcionális):")

# Oldalsó sáv: modulválasztó
st.sidebar.title("Modulválasztó")
module_name = st.sidebar.radio("Kérlek válassz:", list(MODULES.keys()))

# Modul futtatása biztonságosan
safe_run(module_name)
