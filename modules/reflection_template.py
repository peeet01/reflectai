import streamlit as st

def reflection_template_module():
    st.header("🔁 Reflexió sablon")
    st.text_input("Mi történt ma?", key="reflection_today")
    st.text_input("Mit tanultál belőle?", key="reflection_learning")
