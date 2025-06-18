import streamlit as st

from modules.kuramoto_sim import run as run_kuramoto
from modules.hebbian_learning import run as run_hebbian
from modules.hebbian_learning_visual import run as run_hebbian_visual
from modules.kuramoto_hebbiansim import run as run_kuramoto_hebbian
from modules.graph_sync_analysis import run as run_graph_sync
from modules.lorenz_sim import run as run_lorenz
from modules.mlp_predict_lorenz import run as run_mlp_lorenz
from modules.predict_lorenz import run as run_predict_lorenz
from modules.xor_prediction import run as run_xor

st.set_page_config(page_title="ReflectAI", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧠 ReflectAI App</h1>", unsafe_allow_html=True)

menu = st.sidebar.selectbox("Válassz modult", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás",
    "Hebbian vizualizáció",
    "Kuramoto + Hebbian",
    "Gráf szinkron analízis",
    "Lorenz szimuláció",
    "Lorenz előrejelzés MLP-vel",
    "Lorenz előrejelzés (klasszikus)",
    "XOR predikció"
])

if menu == "Kuramoto szinkronizáció":
    run_kuramoto()
elif menu == "Hebbian tanulás":
    run_hebbian()
elif menu == "Hebbian vizualizáció":
    run_hebbian_visual()
elif menu == "Kuramoto + Hebbian":
    run_kuramoto_hebbian()
elif menu == "Gráf szinkron analízis":
    run_graph_sync()
elif menu == "Lorenz szimuláció":
    run_lorenz()
elif menu == "Lorenz előrejelzés MLP-vel":
    run_mlp_lorenz()
elif menu == "Lorenz előrejelzés (klasszikus)":
    run_predict_lorenz()
elif menu == "XOR predikció":
    run_xor()
