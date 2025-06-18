
import streamlit as st

st.set_page_config(page_title="ReflectAI App", layout="wide")
st.title("🧠 ReflectAI App")

menu = st.sidebar.selectbox("Válassz modult:", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás zajjal",
    "XOR predikció",
    "Adaptív Kuramoto–Hebbian háló",
    "Tudományos kérdés: Topológia és zaj hatása"
])

from modules import (
    kuramoto_sim,
    hebbian_learning_vi,
    xor_prediction,
    graph_sync_analysis
)

if menu == "Kuramoto szinkronizáció":
    kuramoto_sim.run()

elif menu == "Hebbian tanulás zajjal":
    hebbian_learning_vi.run()

elif menu == "XOR predikció":
    xor_prediction.run()

elif menu == "Adaptív Kuramoto–Hebbian háló":
    graph_sync_analysis.run(mode="adaptive")

elif menu == "Tudományos kérdés: Topológia és zaj hatása":
    graph_sync_analysis.run(mode="scientific")
