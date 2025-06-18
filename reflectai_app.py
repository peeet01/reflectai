
import streamlit as st
from modules.kuramoto_sim import run as run_kuramoto
from modules.hebbian_learning import run as run_hebbian
from modules.xor_prediction import run as run_xor
from modules.graph_sync_analysis import run as run_graph
from modules.kuramoto_hebbian_sim import run as run_kuramoto_hebbian
from modules.lorenz_sim import run as run_lorenz
from modules.predict_lorenz import run as run_predict_lorenz

st.set_page_config(page_title="ReflectAI Pro", layout="wide")
st.title("ReflectAI Pro – Kutatási MI Platform")

page = st.sidebar.radio("Válassz modult", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás",
    "XOR predikció",
    "Kuramoto–Hebbian adaptív háló",
    "Topológia + zaj hatás",
    "Lorenz attraktor",
    "Lorenz predikció"
])

if page == "Kuramoto szinkronizáció":
    run_kuramoto()
elif page == "Hebbian tanulás":
    run_hebbian()
elif page == "XOR predikció":
    run_xor()
elif page == "Kuramoto–Hebbian adaptív háló":
    run_kuramoto_hebbian()
elif page == "Topológia + zaj hatás":
    run_graph()
elif page == "Lorenz attraktor":
    run_lorenz()
elif page == "Lorenz predikció":
    run_predict_lorenz()
