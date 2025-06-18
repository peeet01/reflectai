import streamlit as st
from modules.kuramoto_sim import run as run_kuramoto
from modules.kuramoto_hebbian_sim import run as run_kuramoto_hebbian
from modules.hebbian_learning import run as run_hebbian
from modules.graph_sync_analysis import run as run_graph_sync
from modules.xor_prediction import run as run_xor
from modules.lorenz_sim import run as run_lorenz_sim
from modules.predict_lorenz import run as run_predict_lorenz

st.set_page_config(page_title="ReflectAI", layout="wide")
st.title("🧠 ReflectAI – Tudományos MI szimulációk")

# Oldalsáv menü
st.sidebar.title("📂 Modulválasztó")
page = st.sidebar.radio("Válassz szimulációs modult", [
    "Kuramoto szinkronizáció",
    "Hebbian tanulás",
    "XOR predikció",
    "Kuramoto-Hebbian adaptív háló",
    "Topológia + zaj hatás",
    "Lorenz attraktor",
    "Lorenz predikció"
])

# Kérdésmező globálisan
st.subheader("❓ Tedd fel kérdésed a modellekhez")
question = st.text_input("Írd be a kérdésed:")

if question:
    st.info("Ez a mező egyelőre csak kijelzésre szolgál – a backend válasz nem aktív.")
    st.write("Kérdés:", question)

# Modulok futtatása
if page == "Kuramoto szinkronizáció":
    run_kuramoto()
elif page == "Hebbian tanulás":
    run_hebbian()
elif page == "XOR predikció":
    run_xor()
elif page == "Kuramoto-Hebbian adaptív háló":
    run_kuramoto_hebbian()
elif page == "Topológia + zaj hatás":
    run_graph_sync()
elif page == "Lorenz attraktor":
    run_lorenz_sim()
elif page == "Lorenz predikció":
    run_predict_lorenz()
