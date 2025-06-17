import streamlit as st
import sys
import os

# 📌 A 'modules' mappa elérési útjának beállítása
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from kuramoto_sim import run_kuramoto_simulation, compute_sync_steps
from hebbian_learning import run_hebbian_learning_with_noise
from xor_prediction import run_xor_prediction

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