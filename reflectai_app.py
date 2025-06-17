import streamlit as st
from modules.kuramoto_sim import run_kuramoto_simulation
from modules.hebbian_learning import run_hebbian_learning_with_noise
from modules.xor_prediction import run_xor_prediction

st.set_page_config(page_title="ReflectAI – Fejlesztett Kutatási Prototípus", page_icon="🧠")
st.title("🧠 ReflectAI – Tudományos szintű MI szimulációk")

# Kérdés bekérése
user_input = st.text_input("Kérdésed vagy kutatási parancsod:")

if user_input:
    st.subheader("💡 Szabályalapú válasz")
    st.write("Ez a verzió a kutatási komponensekre fókuszál. Szimulált, tematikus válasz érkezik.")
    st.markdown("### 🔍 Önreflexió:")
    st.write("A rendszer bővített metrikák alapján működik: tanulási zaj, szinkronizációs idő, predikciós pontosság.")

# 🌐 Kuramoto szinkronizáció
st.header("🌐 Kuramoto szinkronizáció")
fig1, steps_needed = run_kuramoto_simulation()
st.pyplot(fig1)
st.success(f"Szinkronizációs idő: {steps_needed} iteráció")

# 🧬 Hebbian tanulás zajmodellel – interaktív vezérlés
st.header("🧬 Hebbian tanulás zajmodellel")
learning_rate = st.slider("Tanulási ráta", 0.01, 1.0, 0.1, 0.01)
noise_level = st.slider("Zaj szintje", 0.0, 1.0, 0.1, 0.01)
iterations = st.slider("Iterációk száma", 10, 1000, 100, 10)

fig2 = run_hebbian_learning_with_noise(
    learning_rate=learning_rate,
    noise_level=noise_level,
    iterations=iterations
)
st.pyplot(fig2)

# 🧠 XOR predikció
st.header("🧠 XOR predikciós tanulási feladat")
accuracy = run_xor_prediction()
st.success(f"Predikciós pontosság: {accuracy:.2f} %")
