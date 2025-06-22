import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run():
    st.title("📁 Adatfeltöltés modul")

    st.markdown("""
    Tölts fel egy `.csv` vagy `.xlsx` fájlt, amit a sandbox többi modulja is elérhet.
    A feltöltött adatot a rendszer automatikusan eltárolja a memóriában (`st.session_state["uploaded_df"]`), így más analitikai vagy prediktív modul használhatja.
    """)

    uploaded_file = st.file_uploader("Fájl kiválasztása", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state["uploaded_df"] = df
            st.success("✅ Az adat sikeresen betöltve!")

            st.subheader("🔍 Adat előnézet")
            st.dataframe(df.head())

            st.subheader("📊 Alap statisztika")
            st.write(df.describe())

            if df.select_dtypes(include='number').shape[1] >= 2:
                st.subheader("📉 Korrelációs hőtérkép")
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.info("Túl kevés numerikus oszlop a korrelációs mátrixhoz.")

        except Exception as e:
            st.error(f"Hiba a fájl feldolgozásakor: {e}")
    else:
        st.info("📤 Kérlek, tölts fel egy adatfájlt.")
