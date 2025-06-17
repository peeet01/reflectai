import streamlit as st
from modules.kuramoto_sim import run_kuramoto_simulation
from modules.hebbian_learning import run_hebbian_learning

st.set_page_config(page_title="ReflectAI – Kutatási prototípus", page_icon="🧠")

st.title("🧠 ReflectAI – Tudományos neuromorf szimulátor")
st.markdown("Ez a prototípus egy memrisztor–fotonikus neuromorf architektúra szoftveres előképe.")

# 1. Felhasználói bemenet
st.header("📥 Bemenet")
user_input = st.text_input("Kérdésed vagy parancsod (introspektív teszthez):")

# 2. Válasz szimuláció
st.header("🤖 Szimulált MI válasz")
if user_input:
    st.success("„Ez egy szimulált kvázitudatos válasz a kérdésedre.”")

    # 3. Kuramoto szimuláció
    st.header("🔄 Kuramoto szinkronizációs modell")
    st.write("Kuramoto-féle oszcillátorháló szimulációja fáziskoherencia vizsgálathoz.")
    fig = run_kuramoto_simulation()
    st.pyplot(fig)

    # 4. Hebbian tanulási modul
    st.header("🧬 Hebbian tanulás szimuláció")
    st.write("Egy egyszerű Hebbian szabály szerint változó szinaptikus súlyok alakulása.")
    fig2 = run_hebbian_learning()
    st.pyplot(fig2)