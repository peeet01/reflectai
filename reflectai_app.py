import streamlit as st

from modules.kuramoto_sim import run as run_kuramoto
from modules.kuramoto_hebbian_sim import run as run_kuramoto_hebbian
from modules.hebbian_learning import run as run_hebbian
from modules.graph_sync_analysis import run as run_graph_sync
from modules.xor_prediction import run as run_xor
from modules.lorenz_sim import run as run_lorenz_sim
from modules.predict_lorenz import run as run_lorenz_pred

# Főcím
st.title("ReflectAI App")

# Oldalsáv – Modulválasztó
page = st.sidebar.selectbox(
    "Válassz modult",
    [
        "Kuramoto szimuláció",
        "Kuramoto-Hebbian szimuláció",
        "Hebbian tanulás",
        "Gráf szinkronizáció",
        "XOR predikció",
        "Lorenz szimuláció",
        "Lorenz predikció"
    ]
)

# Debug: Megmutatjuk, mit választott
# st.write(f"Kiválasztott modul: {page}")

# A modulok futtatása
if page == "Kuramoto szimuláció":
    st.subheader("Kuramoto szimuláció")
    run_kuramoto()

elif page == "Kuramoto-Hebbian szimuláció":
    st.subheader("Kuramoto-Hebbian szimuláció")
    run_kuramoto_hebbian()

elif page == "Hebbian tanulás":
    st.subheader("Hebbian tanulás")
    run_hebbian()

elif page == "Gráf szinkronizáció":
    st.subheader("Gráf szinkronizáció analízis")
    run_graph_sync()

elif page == "XOR predikció":
    st.subheader("XOR predikció")
    run_xor()

elif page == "Lorenz szimuláció":
    st.subheader("Lorenz szimuláció")
    run_lorenz_sim()

elif page == "Lorenz predikció":
    st.subheader("Lorenz predikció")
    run_lorenz_pred()
