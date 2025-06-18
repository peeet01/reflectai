import streamlit as st

from modules.kuramoto_sim import run as run_kuramoto
from modules.kuramoto_hebbian import run as run_kuramoto_hebbian
from modules.hebbian_learning import run as run_hebbian
from modules.graph_sync_analysis import run as run_graph_sync
from modules.xor_prediction import run as run_xor

st.title("ReflectAI App")

page = st.sidebar.selectbox(
    "Válassz modult",
    ["Kuramoto", "Kuramoto-Hebbian", "Hebbian", "Sync Analízis", "XOR Predikció"]
)

if page == "Kuramoto":
    run_kuramoto()
elif page == "Kuramoto-Hebbian":
    run_kuramoto_hebbian()
elif page == "Hebbian":
    run_hebbian()
elif page == "Sync Analízis":
    run_graph_sync()
elif page == "XOR Predikció":
    run_xor()
