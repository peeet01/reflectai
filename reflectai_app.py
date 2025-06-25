import streamlit as st
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from utils.metadata_loader import load_metadata
from modules.modules_registry import MODULES

# --- Konfiguráció betöltése ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# --- Autentikáció ---
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days'],
    preauthorized=config.get('preauthorized', {})
)

# --- Bejelentkezés ---
name, auth_status, username = authenticator.login("main")

# --- Hitelesítési állapot kezelése ---
if auth_status is False:
    st.error("❌ Hibás felhasználónév vagy jelszó.")
elif auth_status is None:
    st.warning("⚠️ Kérlek jelentkezz be.")
elif auth_status:
    st.set_page_config(page_title="Neurolab AI – Scientific Reflection", layout="wide")
    st.sidebar.success(f"✅ Bejelentkezve mint: {name} ({username})")

    st.title("🧠 Neurolab AI – Scientific Reflection")
    st.markdown("Válassz egy modult a bal oldali menüből.")

    # Modulválasztó
    st.sidebar.title("📂 Modulválasztó")
    module_key = st.sidebar.radio("Modul kiválasztása:", list(MODULES.keys()))

    # Metaadat input
    st.text_input("📝 Megfigyelés vagy jegyzet címe:", key="metadata_title")

    # Metaadat betöltés
    metadata_key = module_key.replace(" ", "_").lower()
    metadata = load_metadata(metadata_key)
    st.write("📄 Modul metaadatai:", metadata)

    # Modul futtatása
    selected_func = MODULES.get(module_key)
    if selected_func:
        selected_func()
    else:
        st.error("⚠️ A kiválasztott modul nem található.")
