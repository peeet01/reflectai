import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from modules.journal import journal_module
from modules.reflection_template import reflection_template_module
from utils.metadata_loader import load_metadata  # <-- ez a helyes útvonal!

# 🔐 Konfiguráció betöltése
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    cookie_key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days']
)

# 🔐 Bejelentkezés
name, authentication_status, username = authenticator.login("Login", "main")  # << FONTOS: form_name helyett pozíciós!

if authentication_status is False:
    st.error("Hibás felhasználónév vagy jelszó.")
elif authentication_status is None:
    st.warning("Kérlek add meg a bejelentkezési adataidat.")
elif authentication_status:
    authenticator.logout("Kijelentkezés", "sidebar")
    st.sidebar.success(f"Bejelentkezve mint {name}")

    # Oldalválasztó
    st.sidebar.title("Navigáció")
    page = st.sidebar.selectbox("Válassz oldalt", ["Kutatási napló", "Reflexió sablon"])

    MODULES = {
        "Kutatási napló": journal_module,
        "Reflexió sablon": reflection_template_module,
    }

    # Modul betöltése
    if page in MODULES:
        MODULES[page]()

    # Metaadatok
    metadata = load_metadata(page)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Verzió:** {metadata['version']}")
    st.sidebar.markdown(f"**Fejlesztő:** {metadata['author']}")
