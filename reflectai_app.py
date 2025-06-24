import streamlit as st
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from utils.metadata_loader import load_metadata

# --- Konfiguráció betöltése ---
with open("config.yaml") as file:
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
name, authentication_status, username = authenticator.login("Bejelentkezés", location="main")

if authentication_status is False:
    st.error("❌ Hibás felhasználónév vagy jelszó.")
elif authentication_status is None:
    st.warning("⚠️ Kérlek jelentkezz be.")
else:
    st.set_page_config(page_title="Neurolab AI – Scientific Reflection", layout="wide")
    st.sidebar.success(f"✅ Bejelentkezve mint: {name} ({username})")
    authenticator.logout("Kijelentkezés", location="sidebar")

    st.title("🧠 Neurolab AI – Scientific Reflection")
    st.markdown("Válassz egy modult a bal oldali menüből.")

    st.sidebar.title("📂 Modulválasztó")
    module_key = st.sidebar.radio("Kérlek válassz modult:", ["Kutatási napló", "Reflexió sablon"])
    metadata_key = module_key.replace(" ", "_").lower()

    st.text_input("📝 Megfigyelés vagy jegyzet címe:", key="metadata_title")

    metadata = load_metadata(metadata_key)
    st.write("📄 Modul metaadatai:", metadata)
