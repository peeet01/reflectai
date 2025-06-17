import streamlit as st

st.set_page_config(page_title="ReflectAI DEMÓ", page_icon="🧠")
st.markdown("## 🧠 ReflectAI – Kvázitudati MI (DEMO mód)")
st.info("🔧 Ez a demó mód aktív, nincs OpenAI API kulcs beállítva. A válasz szimulált.")

user_input = st.text_input("Kérdésed vagy feladatod:")

if user_input:
    with st.spinner("Szimulált válasz generálása..."):
        st.markdown("### 🤖 Válasz:")
        st.write(f'"{user_input}" kérdésedre válaszként ezt gondolom:')
        st.write("Ez egy szimulált válasz, amit a DEMO rendszer állított elő.")

        st.markdown("### 🔍 Önreflexió:")
        st.write("A válaszom szerintem koherens, a kérdésed tartalmára összpontosít. "
                 "Bár nincs valódi nyelvi modell a háttérben, a rendszer képes lehet "
                 "önreflexív modul integrálására a jövőben.")