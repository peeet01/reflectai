import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from modules.modules_registry import MODULES, safe_run
from utils.metadata_loader import load_metadata

# Oldalbeállítás
st.set_page_config(page_title="ReflectAI – Scientific Sandbox", layout="wide")

# --- Konfiguráció betöltése ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# --- Autentikáció beállítása ---
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    expiry_days=config['cookie']['expiry_days'],
    preauthorized=config.get('preauthorized', {})
)

# --- Bejelentkezés ---
name, authentication_status, username = authenticator.login('Bejelentkezés', location='main')

# --- Hitelesítési állapot kezelése ---
if authentication_status is False:
    st.error("❌ Hibás felhasználónév vagy jelszó.")
elif authentication_status is None:
    st.warning("⚠️ Kérlek jelentkezz be.")
elif authentication_status:
    authenticator.logout("Kijelentkezés", location='sidebar')
    st.sidebar.success(f"✅ Bejelentkezve mint: {name} ({username})")

    # Főcím
    st.title("🧠 ReflectAI – Scientific Playground")

    # Modulválasztó
    st.sidebar.title("🗂️ Modulválasztó")
    selected_module = st.sidebar.radio("Válassz modult:", list(MODULES.keys()))

    # Metaadat betöltés és megjelenítés
    metadata_key = selected_module.replace(" ", "_").lower()
    metadata = load_metadata(metadata_key)

    st.subheader(f"📘 {metadata.get('title', selected_module)}")
    st.write(metadata.get("description", ""))
