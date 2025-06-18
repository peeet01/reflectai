
def run(mode='adaptive'):
    import streamlit as st
    if mode == 'adaptive':
        st.header('Adaptív háló modul fut')
    elif mode == 'scientific':
        st.header('Tudományos kérdés modul fut')
