import streamlit as st
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from utils.metadata_loader import load_metadata

# Oldal beállítások
st.set_page_config(page_title="Neurolab AI – Scientific Reflection", layout="wide")

# --- Konfiguráció betöltése ---
with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=SafeLoader)

# --- Autentikáció beállítása ---
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days'],
    preauthorized=config['preauthorized']
)

# --- Bejelentkezés ---
name, authentication_status, username = authenticator.login("main", "Bejelentkezés")

# --- Hitelesítési állapot kezelése ---
if authentication_status is False:
    st.error("❌ Hibás felhasználónév vagy jelszó.")
elif authentication_status is None:
    st.warning("⚠️ Kérlek jelentkezz be.")
else:
    # Sikeres bejelentkezés után
    st.sidebar.success(f"✅ Bejelentkezve mint: {name} ({username})")

    st.title("🧠 Neurolab AI – Scientific Reflection")
    st.markdown("Válassz egy modult a bal oldali menüből.")

    # --- Modulválasztó ---
    st.sidebar.title("📂 Modulválasztó")
    module_key = st.sidebar.radio("Kérlek válassz modult:", ["Kutatási napló", "Reflexió sablon"])

    # --- Metaadat bekérés ---
    st.text_input("📝 Megfigyelés vagy jegyzet címe:", key="metadata_title")

    # --- Metaadat betöltés ---
    metadata_key = module_key.replace(" ", "_").lower()
    metadata = load_metadata(metadata_key)
    st.subheader("📄 Modul metaadatai")
    st.json(metadata)
