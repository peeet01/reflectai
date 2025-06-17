
import streamlit as st
from modules.kuramoto_sim import run_kuramoto_simulation
from modules.hebbian_learning import run_hebbian_learning_with_noise
from modules.xor_prediction import run_xor_prediction
from modules.kuramoto_hebbian import run_kuramoto_hebbian

st.set_page_config(page_title="ReflectAI – Fejlesztett Kutatási Prototípus", page_icon="🧠")
st.title("🧠 ReflectAI – Tudományos szintű MI szimulációk")

user_input = st.text_input("Kérdésed vagy kutatási parancsod:")
if user_input:
    st.subheader("💡 Szabályalapú válasz")
    st.write("Ez a verzió a kutatási komponensekre fókuszál. Szimulált, tematikus válasz érkezik.")
    st.markdown("### 🔍 Önreflexió:")
    st.write("A rendszer bővített metrikák alapján működik: tanulási zaj, szinkronizációs idő, predikciós pontosság.")

# Kuramoto metrika
st.header("🌐 Kuramoto szinkronizáció")
fig1, steps_needed = run_kuramoto_simulation()
st.pyplot(fig1)
st.success(f"Szinkronizációs idő: {steps_needed} iteráció")

# Hebbian tanulás zajjal
st.header("🧬 Hebbian tanulás zajmodellel")
fig2 = run_hebbian_learning_with_noise()
st.pyplot(fig2)

# XOR predikció
st.header("🧠 XOR predikciós tanulási feladat")
accuracy = run_xor_prediction()
st.success(f"Predikciós pontosság: {accuracy:.2f} %")

# Kuramoto–Hebbian topologikus szimuláció
st.header("🔁 Adaptív Kuramoto–Hebbian háló")
fig4, topo_stats = run_kuramoto_hebbian()
st.pyplot(fig4)
st.success(f"Topológiai koherencia: {topo_stats['coherence']:.2f}, Szinkron iteráció: {topo_stats['sync_steps']}")
