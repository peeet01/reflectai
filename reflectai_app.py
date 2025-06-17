import streamlit as st
from modules.kuramoto_sim import run_kuramoto_simulation
from modules.hebbian_learning import run_hebbian_learning

st.set_page_config(page_title="ReflectAI – Okos válaszoló", page_icon="🧠")
st.title("🧠 ReflectAI – Kulcsszó-alapú válaszoló rendszer")

# 1. Felhasználói bemenet
st.header("📥 Bemenet")
user_input = st.text_input("Írd be a kérdésed vagy feladatod:")

def generate_keyword_based_response(text):
    text = text.lower()
    if "introspekció" in text or "önreflexió" in text:
        return ("Az introspekció egy önmegfigyelési folyamat, "
                "ahol a rendszer képes saját válaszainak és belső állapotainak értékelésére."), (
                "Ez a válasz kapcsolódik a kérdésedhez, és tükrözi az MI introspektív működésének elvét.")
    elif "tanulás" in text:
        return ("A rendszer képes szinaptikus súlyok módosítására Hebbian tanulás szerint, "
                "ami biológiailag ihletett mechanizmus."), (
                "A tanulás leírása illeszkedik a kérdéshez, és a rendszer funkciójára utal.")
    elif "neurális" in text or "memrisztor" in text:
        return ("A memrisztor-alapú neurális hálók képesek analóg tanulásra és alacsony energiafelhasználásra."), (
                "A válasz releváns, de további részletezéssel pontosítható lenne.")
    else:
        return ("A kérdésed érdekes, de a rendszer jelenlegi tudásával nem tud specifikus választ adni."), (
                "A válasz általános jellegű, de további fejlesztéssel specifikálható.")

# 2. Válasz és introspekció
if user_input:
    st.subheader("💡 Válasz a kérdésedre:")
    answer, reflection = generate_keyword_based_response(user_input)
    st.write(answer)

    st.markdown("### 🔍 Önreflexió:")
    st.write(reflection)

    # 3. Kuramoto szimuláció
    st.header("🌐 Kuramoto szinkronizációs szimuláció")
    fig = run_kuramoto_simulation()
    st.pyplot(fig)

    # 4. Hebbian tanulási szimuláció
    st.header("🧬 Hebbian tanulás szimuláció")
    fig2 = run_hebbian_learning()
    st.pyplot(fig2)