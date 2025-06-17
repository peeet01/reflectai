import streamlit as st
from modules.kuramoto_sim import run_kuramoto_simulation
from modules.hebbian_learning import run_hebbian_learning

st.set_page_config(page_title="ReflectAI – Szabályalapú válaszoló", page_icon="🧠")
st.title("🧠 ReflectAI – Kvázitudatos szabályalapú MI")

# 1. Felhasználói bemenet
st.header("📥 Bemenet")
user_input = st.text_input("Írd be a kérdésed vagy feladatod:")

# 2. Okos szabálymotor
def generate_response(text):
    text = text.lower()
    if any(q in text for q in ["hogyan", "működik", "működése"]):
        if "introspekció" in text:
            return ("Az introspekció működése az MI-ben azt jelenti, hogy a rendszer képes felismerni saját válaszainak logikai szerkezetét és hibáit."), "A válasz kifejezetten a működésre fókuszál, ez kontextuálisan pontos."
        if "tanulás" in text:
            return ("A tanulás folyamata Hebbian elvek alapján történik, ahol a szinapszis erősödik, ha a bemenet és a válasz együtt aktiválódik."), "A válasz illeszkedik a kérdés szerkezetéhez és a kulcskifejezésekhez."
    elif any(q in text for q in ["mi az", "mi az a", "definiáld", "fogalma"]):
        if "introspekció" in text:
            return ("Az introspekció a mesterséges intelligenciában a rendszer önmegfigyelési és önértékelési képességét jelenti."), "A válasz definíciószerű, releváns és kontextusfüggő."
        if "kuramoto" in text:
            return ("A Kuramoto-modell egy szinkronizációs elmélet, mely oszcillátorok fázisait modellezi, például a fotonikus hálózatokban."), "A válasz helyesen foglalja össze a Kuramoto-modellt."
    elif "memrisztor" in text:
        return ("A memrisztor egy olyan nanoszerkezet, amely képes ellenállás változással tárolni információt – szinaptikus súlyként használható."), "A válasz technikailag korrekt és informatív."

    return ("A kérdésed összetett, de a jelenlegi szabályalapú válaszadó nem talál pontos egyezést.", "A válasz általános, a rendszer fejlesztése javasolt a további finomhangoláshoz.")

# 3. Válasz és introspekció
if user_input:
    st.subheader("💡 Válasz a kérdésedre:")
    answer, reflection = generate_response(user_input)
    st.write(answer)
    st.markdown("### 🔍 Önreflexió:")
    st.write(reflection)

    st.header("🌐 Kuramoto szinkronizációs szimuláció")
    fig = run_kuramoto_simulation()
    st.pyplot(fig)

    st.header("🧬 Hebbian tanulás szimuláció")
    fig2 = run_hebbian_learning()
    st.pyplot(fig2)