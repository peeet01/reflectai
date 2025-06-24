import streamlit as st
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from utils import load_metadata  # Gondoskodj róla, hogy ez létezik

# Oldal beállítás
st.set_page_config(page_title="Neurolab AI - Scientific Reflection", layout="wide")

# --- Beállítások betöltése ---
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# --- Autentikáció beállítása ---
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# --- Bejelentkezés ---
name, authentication_status, username = authenticator.login("main", "Bejelentkezés")

# --- Hitelesítési logika ---
if authentication_status is False:
    st.error("Hibás felhasználónév vagy jelszó.")
elif authentication_status is None:
    st.warning("Kérlek jelentkezz be a folytatáshoz.")
elif authentication_status:
    st.sidebar.success(f"Bejelentkezve: {name} ({username})")

    st.title("🧠 Neurolab AI – Scientific Reflection")
    st.markdown("Válassz egy modult a bal oldali menüből.")

    # Modulválasztó
    st.sidebar.title("📂 Modulválasztó")
    module_key = st.sidebar.radio("Kérlek válassz modult:", ("Kutatási napló", "Reflexió sablon"))

    # Metaadat bekérés
    st.text_input("📝 Megfigyelés vagy jegyzet címe:", key="metadata_title")

    # Metaadat betöltés (mock példa)
    metadata_key = module_key.replace(" ", "_").lower()
    metadata = load_metadata(metadata_key)
    st.write("🔍 Modul metaadatai:", metadata)
