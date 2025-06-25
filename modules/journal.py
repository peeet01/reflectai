import streamlit as st

def journal_module():
    st.header("📔 Kutatási napló")
    st.text_area("Jegyzetek", key="journal_notes")
