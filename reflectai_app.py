
import streamlit as st
from modules.kuramoto_sim import run_kuramoto_simulation
from modules.hebbian_learning import run_hebbian_learning_with_noise
from modules.xor_prediction import run_xor_prediction
from modules.kuramoto_hebbian import run_kuramoto_hebbian
from modules.graph_sync_analysis import compare_graph_topologies

st.set_page_config(page_title="ReflectAI – Tudományos MI Prototípus", page_icon="🧠")
st.title("🧠 ReflectAI – Tudományos MI szimulációk")

user_input = st.text_input("Kérdésed vagy kutatási parancsod:")
if user_input:
    st.subheader("💡 Elemzés generálása")
    st.write("A rendszer jelenleg a fő szinkronizációs és tanulási modulokra reagál.")

# 1. Kuramoto
st.header("🌐 Kuramoto szinkronizáció")
fig1, steps_needed = run_kuramoto_simulation()
st.pyplot(fig1)
st.success(f"Szinkronizációs idő: {steps_needed} iteráció")

# 2. Hebbian
st.header("🧬 Hebbian tanulás zajjal")
fig2 = run_hebbian_learning_with_noise()
st.pyplot(fig2)

# 3. XOR tanulás
st.header("🧠 XOR predikciós tanulási feladat")
accuracy = run_xor_prediction()
st.success(f"Predikciós pontosság: {accuracy:.2f} %")

# 4. Kuramoto–Hebbian dinamikus háló
st.header("🔁 Adaptív Kuramoto–Hebbian háló")
fig4, topo_stats = run_kuramoto_hebbian()
st.pyplot(fig4)
st.success(f"Topológiai koherencia: {topo_stats['coherence']:.2f}, Szinkron iteráció: {topo_stats['sync_steps']}")

# 5. Tudományos kérdés szimuláció
st.header("🧪 Tudományos kérdés: Topológia és zaj hatása")
fig_demo, demo_results = compare_graph_topologies()
st.pyplot(fig_demo)
for gtype, t in demo_results.items():
    st.info(f"{gtype} gráf szinkronizációs ideje: {t} iteráció")
