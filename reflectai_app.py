import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from modules.journal import journal_module
from modules.reflection_template import reflection_template_module
from utils.metadata_loader import load_metadata

# Konfiguráció betöltése
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    cookie_key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days'],
    preauthorized=config.get('preauthorized', {})
)

name, authentication_status, username = authenticator.login("Bejelentkezés", "main")

if authentication_status is False:
    st.error("❌ Hibás felhasználónév vagy jelszó.")
elif authentication_status is None:
    st.warning("⚠️ Kérlek jelentkezz be.")
elif authentication_status:
    authenticator.logout("Kijelentkezés", "sidebar")
    st.sidebar.success(f"✅ Bejelentkezve mint: {name}")

    st.title("🧠 ReflectAI – Scientific Reflection")
    st.markdown("Válassz egy modult a bal oldali menüből.")

    page = st.sidebar.selectbox("📂 Modul kiválasztása", ["Kutatási napló", "Reflexió sablon"])
    MODULES = {
        "Kutatási napló": journal_module,
        "Reflexió sablon": reflection_template_module,
    }

    if page in MODULES:
        MODULES[page]()  # modul meghívása

    metadata = load_metadata(page)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Verzió:** {metadata.get('version', '1.0')}")
    st.sidebar.markdown(f"**Fejlesztő:** {metadata.get('author', 'ReflectAI')}")
