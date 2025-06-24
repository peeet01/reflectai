import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from modules.modules_registry import MODULES, safe_run
from utils.metadata_loader import load_metadata

# --- Alapbeállítások ---
st.set_page_config(page_title="Neurolab AI – Scientific Reflection", page_icon="🧠", layout="wide")

# --- Konfiguráció betöltése ---
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# --- Authenticator példányosítás ---
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days'],
    preauthorized=config.get('preauthorized', {})
)

# --- Bejelentkezés ---
auth_result = authenticator.login("main", "Bejelentkezés")

# --- Felhasználó státusz kezelése ---
if auth_result is None:
    st.warning("⚠️ Kérlek jelentkezz be.")
elif not auth_result['authenticated']:
    st.error("❌ Hibás felhasználónév vagy jelszó.")
else:
    name = auth_result['name']
    username = auth_result['username']

    st.sidebar.success(f"✅ Bejelentkezve mint: {name} ({username})")
    st.title("🧠 Neurolab AI – Scientific Playground Sandbox")
    st.markdown("Válassz egy modult a bal oldali sávból a vizualizáció indításához.")
    st.text_input("📝 Megfigyelés vagy jegyzet (opcionális):")

    # Modulválasztó
    st.sidebar.title("🗂️ Modulválasztó")
    module_key = st.sidebar.radio("Kérlek válassz egy modult:", list(MODULES.keys()))

    # Metaadat betöltés
    metadata_key = module_key.replace(" ", "_").lower()
    metadata = load_metadata(metadata_key)

    # Metaadatok megjelenítése
    st.subheader(f"📘 {metadata.get('title', module_key)}")
    st.write(metadata.get("description", ""))

    if metadata.get("equations"):
        st.markdown("#### 🧮 Egyenletek:")
        for eq in metadata["equations"]:
            st.latex(eq)

    if metadata.get("parameters"):
        st.markdown("#### 🎛️ Paraméterek:")
        for param, desc in metadata["parameters"].items():
            st.markdown(f"- **{param}**: {desc}")

    if metadata.get("applications"):
        st.markdown("#### 🔬 Alkalmazási területek:")
        for app in metadata["applications"]:
            st.markdown(f"- {app}")

    st.divider()

    # Modul futtatás
    safe_run(module_key)
