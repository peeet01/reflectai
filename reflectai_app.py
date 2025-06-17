import streamlit as st
from modules.kuramoto_sim import run_kuramoto_simulation

st.set_page_config(page_title="ReflectAI – Fejlesztett Kutatási Prototípus", page_icon="🧠")
st.title("🧠 ReflectAI – Tudományos szintű MI szimulációk")

user_input = st.text_input("Kérdésed vagy kutatási parancsod:")

if user_input:
    st.subheader("💡 Szabályalapú válasz")
    st.write("Ez a verzió a kutatási komponensekre fókuszál. Szimulált, tematikus válasz érkezik.")
    st.markdown("### 🔍 Önreflexió:")
    st.write("A rendszer bővített metrikák alapján működik: tanulási zaj, szinkronizációs idő, predikciós pontosság.")

st.header("🌐 Kuramoto szinkronizáció")
fig1, steps_needed = run_kuramoto_simulation()
st.pyplot(fig1)
st.success(f"Szinkronizációs idő: {steps_needed} iteráció")