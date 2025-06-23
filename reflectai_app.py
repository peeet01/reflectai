
import streamlit as st
from modules.persistent_homology import run as run_homology
from modules.reflection_modul import run as run_reflection

st.set_page_config(page_title="Neurolab AI – Scientific Playground Sandbox", page_icon="🧠", layout="wide")
st.title("Neurolab AI – Scientific Playground Sandbox")
st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció indításához.")

st.sidebar.title("Modulválasztó")
module_name = st.sidebar.radio("Kérlek válassz:", (
    "Perzisztens homológia",
    "Napi önreflexió"
))

if module_name == "Perzisztens homológia":
    run_homology()
elif module_name == "Napi önreflexió":
    run_reflection()
