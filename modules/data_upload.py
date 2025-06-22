
import streamlit as st
import pandas as pd

def run():
    st.title("📁 Adatfeltöltés modul – Neurolab AI")
    st.markdown("""
    Ez a modul lehetővé teszi saját CSV adatok feltöltését és előnézetét. Az adatok később más modulokban is felhasználhatók.
    """)

    uploaded_file = st.file_uploader("📤 Tölts fel egy CSV fájlt", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Fájl sikeresen beolvasva!")

            st.markdown("### 📊 Előnézet az adatokról")
            st.dataframe(df)

            st.markdown("### 📈 Alap statisztikák")
            st.write(df.describe())

            # Használhatóság más modulokban
            st.session_state["uploaded_data"] = df

            st.markdown("✅ Az adatok elérhetők más modulokból `st.session_state['uploaded_data']` formában.")

        except Exception as e:
            st.error(f"Hiba történt a fájl feldolgozásakor: {e}")
