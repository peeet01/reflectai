import streamlit as st
import pandas as pd
from io import StringIO

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Hiba a fájl beolvasásakor: {e}")
    return None

def get_uploaded_data():
    st.sidebar.subheader("📁 Adatfeltöltés")
    uploaded_file = st.sidebar.file_uploader("Tölts fel egy CSV fájlt", type=["csv"])

    df = load_data(uploaded_file)

    if uploaded_file is not None:
        st.sidebar.success("✅ Fájl betöltve")
    else:
        st.sidebar.info("📂 Várakozás fájl feltöltésére...")

    return df

def show_data_overview(df, title="📊 Feltöltött adat előnézete"):
    if df is not None:
        st.subheader(title)
        st.write("ℹ️ Adatok mérete:", df.shape)
        st.dataframe(df.head())

        if df.isnull().values.any():
            st.warning("⚠️ Hiányzó értékek találhatók az adathalmazban!")
    else:
        st.info("📂 Nincs feltöltött adat.")

def run():
    st.title("📁 Adatfeltöltés modul")
    st.markdown("""
    Ez a modul lehetővé teszi CSV fájlok feltöltését és az adatok gyors áttekintését.
    Legalább 3 oszlopos idősor ajánlott a további feldolgozáshoz.
    """)
    df = get_uploaded_data()
    show_data_overview(df)
