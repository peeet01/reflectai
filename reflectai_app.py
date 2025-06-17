
import streamlit as st
import openai
from openai import OpenAI

st.set_page_config(page_title="ReflectAI", page_icon="🧠")

st.markdown("## 🧠 ReflectAI – Kvázitudati MI prototípus (OpenAI + introspekció)")
st.write("Írd be a kérdésed, és a rendszer válaszol + önreflexióval értékeli magát.")

api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("Hiányzik az OPENAI_API_KEY. Állítsd be a Streamlit Secrets-ben vagy .env-ben.")
    st.stop()

client = OpenAI(api_key=api_key)

user_input = st.text_input("Kérdésed vagy feladatod:")

if user_input:
    with st.spinner("Válasz generálása..."):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Te egy introspektív MI vagy. Válaszolj, majd értékeld is a saját válaszod."},
                    {"role": "user", "content": user_input}
                ]
            )
            answer = response.choices[0].message.content
            st.markdown("### 🤖 Válasz:")
            st.write(answer)
        except Exception as e:
            st.error(f"Válaszgenerálás hiba: {e}")
