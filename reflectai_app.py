import streamlit as st
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from utils import load_metadata  # biztosítsd, hogy ez a függvény létezik

# --- Beállítások betöltése ---
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# --- Autentikáció beállítása ---
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- Bejelentkezés ---
name, auth_status, username, _ = authenticator.login("main", "Bejelentkezés")

# --- Hitelesítés kezelése ---
if auth_status is False:
    st.error("Hibás felhasználónév vagy jelszó.")
elif auth_status is None:
    st.warning("Kérlek jelentkezz be.")
elif auth_status:
    st.sidebar.success(f"Bejelentkezve mint: {name} ({username})")

    # Oldal beállítások
    st.set_page_config(page_title="Neurolab AI - Scientific Reflection", layout="wide")
    st.title("🧠 Neurolab AI – Scientific Reflection")
    st.markdown("Válassz egy modult a bal oldali menüből.")

    # Modulválasztó
    st.sidebar.title("📂 Modulválasztó")
    module_key = st.sidebar.radio("Kérlek válassz modult:", ("Kutatási napló", "Reflexió sablon"))

    # Metaadat bekérés
    st.text_input("📝 Megfigyelés vagy jegyzet címe:", key="metadata_title")

    # Metaadat betöltés (dummy logika példa)
    metadata_key = module_key.replace(" ", "_").lower()
    metadata = load_metadata(metadata_key)
    st.write("🔍 Modul metaadatai:", metadata)
