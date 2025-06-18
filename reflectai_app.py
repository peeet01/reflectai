
import streamlit as st
from modules import kuramoto_sim, hebbian_learning, xor_prediction, graph_sync_analysis

st.set_page_config(page_title="ReflectAI App", layout="wide")
st.title("🧠 ReflectAI App")

menu = st.sidebar.radio("Modul választása", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás zajjal",
    "XOR predikció",
    "Adaptív Kuramoto–Hebbian háló",
    "Tudományos kérdés: Topológia és zaj hatása"
])

if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()
elif menu == "Hebbian tanulás zajjal":
    hebbian_learning.run()
elif menu == "XOR predikció":
    xor_prediction.run()
elif menu == "Adaptív Kuramoto–Hebbian háló":
    graph_sync_analysis.run_adaptive_sync()
elif menu == "Tudományos kérdés: Topológia és zaj hatása":
    graph_sync_analysis.run_topology_noise()
else:
    st.warning("Ismeretlen modul.")
