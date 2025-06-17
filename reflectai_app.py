import streamlit as st
import random
from modules.kuramoto_sim import run_kuramoto_simulation
from modules.hebbian_learning import run_hebbian_learning

st.set_page_config(page_title="ReflectAI – DEMÓ válaszoló", page_icon="🧠")
st.title("🧠 ReflectAI – Kvázi-introspektív válaszoló rendszer")

# 1. Felhasználói bemenet
st.header("📥 Bemenet")
user_input = st.text_input("Írd be a kérdésed vagy feladatod:")

# 2. Szimulált válasz és introspekció
if user_input:
    st.subheader("💡 Válasz a kérdésedre:")

    fake_answers = [
        "Ez egy szimulált válasz, amit a DEMO rendszer állított elő.",
        "A kérdésed releváns, és a rendszer feltételezi, hogy az introspekció az önmegfigyelés aktusa.",
        "A mesterséges intelligencia introspektív képességei jelenleg szimuláció szintjén léteznek."
    ]

    fake_reflections = [
        "A válaszom szerintem koherens, a kérdésed tartalmára összpontosít.",
        "A válasz lehetséges, de szükséges lenne mélyebb nyelvi megértés.",
        "A rendszer jelenleg csak sablonos önreflexióval dolgozik, de bővíthető."
    ]

    st.write(random.choice(fake_answers))
    st.markdown("### 🔍 Önreflexió:")
    st.write(random.choice(fake_reflections))

    # 3. Kuramoto szimuláció
    st.header("🌐 Kuramoto szinkronizációs szimuláció")
    fig = run_kuramoto_simulation()
    st.pyplot(fig)

    # 4. Hebbian tanulási szimuláció
    st.header("🧬 Hebbian tanulás szimuláció")
    fig2 = run_hebbian_learning()
    st.pyplot(fig2)