
import streamlit as st

def journal_module():
    st.header("📔 Kutatási napló")
    st.markdown("Itt vezetheted a kutatási megfigyeléseidet.")
    st.text_area("Jegyzet", key="journal_notes")
