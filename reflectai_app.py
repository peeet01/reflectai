
import streamlit as st
import openai
import os
import random
import json
from dotenv import load_dotenv
from datetime import datetime
import matplotlib.pyplot as plt

# === Konfigurációk ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
PSI_FILE = "psi_memory.json"

# === Psi_t állapotkezelés ===
def build_internal_state(confidence):
    state = {
        "confidence": confidence,
        "coherence": round(random.uniform(0.7, 0.95), 2),
        "timestamp": datetime.now().isoformat()
    }
    return state

def save_state_to_file(state):
    if not os.path.exists(PSI_FILE):
        with open(PSI_FILE, "w") as f:
            json.dump([], f)
    with open(PSI_FILE, "r+") as f:
        data = json.load(f)
        data.append(state)
        f.seek(0)
        json.dump(data, f, indent=2)

def generate_reflection(confidence):
    if confidence >= 85:
        return "Magabiztos vagyok a válaszomban, mert jól illeszkedik a kérdéshez és releváns információt tartalmaz."
    elif 70 <= confidence < 85:
        return "Viszonylag biztos vagyok a válaszban, de előfordulhat, hogy hiányzik néhány fontos részlet."
    else:
        return "Ez a válasz bizonytalan. Elképzelhető, hogy nem tartalmaz minden releváns információt vagy nem teljesen pontos."

def evaluate_with_openai(question, answer):
    prompt = f"""Feladat: Értékeld az alábbi MI válasz minőségét 0-tól 100-ig a következő szempontok alapján:
- Mennyire pontos?
- Mennyire összefüggő?
- Mennyire biztosnak tűnik?

Kérdés: {question}
Válasz: {answer}

Kérlek, csak egy számot adj vissza!

Értékelés:
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response['choices'][0]['message']['content']
        digits = ''.join(filter(str.isdigit, content))
        score = int(digits[:3])
        return min(score, 100)
    except Exception as e:
        st.error(f"Hiba az OpenAI API-val: {e}")
        return 50

def generate_alternative(question):
    alternatives = [
        "Egyes nézőpontok szerint más okok is felmerülhetnek...",
        "Alternatív válasz: gazdasági tényezők is szerepet játszottak...",
        "Más nézőpontból a kérdés árnyaltabban értelmezhető..."
    ]
    return random.choice(alternatives)

# === Streamlit UI ===
st.set_page_config(page_title="ReflectAI - Kvázitudati MI", page_icon="🧠")
st.title("🧠 ReflectAI – Kvázitudati MI prototípus (OpenAI + introspekció)")
st.markdown("Írd be a kérdésed, és a rendszer válaszol + önreflexióval értékeli magát.")

user_input = st.text_input("Kérdésed vagy feladatod:")

if user_input:
    with st.spinner("Válasz generálása és introspektív értékelés..."):

        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=300,
                messages=[{"role": "user", "content": user_input}]
            )
            response = gpt_response['choices'][0]['message']['content'].strip()
        except Exception as e:
            st.error(f"Válaszgenerálás hiba: {e}")
            response = "Hiba történt a válasz generálása során."

        confidence = evaluate_with_openai(user_input, response)
        psi_t = build_internal_state(confidence)
        save_state_to_file(psi_t)

        reflection = generate_reflection(confidence)
        alt_response = generate_alternative(user_input) if confidence < 70 else None

    st.subheader("🤖 Válaszom:")
    st.write(response)

    st.subheader("🧠 Introspektív értékelés:")
    st.write(f"- **Magabiztosság**: {confidence}%")
    st.write(f"- **Reflexió**: {reflection}")

    if alt_response:
        st.subheader("💡 Alternatív nézőpont:")
        st.write(alt_response)

    st.markdown("---")
    st.caption("ReflectAI – Kvázitudati MI prototípus | Állapotmentés: `psi_memory.json`")

    if os.path.exists(PSI_FILE):
        with open(PSI_FILE, "r") as f:
            psi_data = json.load(f)

        if psi_data:
            timestamps = [datetime.fromisoformat(d["timestamp"]) for d in psi_data]
            confidences = [d["confidence"] for d in psi_data]
            coherences = [d["coherence"] for d in psi_data]

            st.subheader("📊 Introspektív állapotok időben")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(timestamps, confidences, label="Magabiztosság (%)", marker='o', color='blue')
            ax.plot(timestamps, coherences, label="Koherencia", marker='s', color='green')
            ax.set_xlabel("Idő")
            ax.set_ylabel("Érték")
            ax.set_title("ReflectAI introspektív állapotai")
            ax.legend()
            ax.grid(True)
            fig.autofmt_xdate()
            st.pyplot(fig)
