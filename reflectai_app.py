import streamlit as st
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader

from modules.modules_registry import MODULES, safe_run

# ⬇️ Betöltjük a konfigurációt
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# ⬇️ Hitelesítő inicializálása
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# ⬇️ Bejelentkezés mezők
name, authentication_status, username = authenticator.login('Bejelentkezés', 'main')

if authentication_status is False:
    st.error('Hibás felhasználónév vagy jelszó')

elif authentication_status is None:
    st.warning('Kérlek jelentkezz be az alkalmazás használatához.')

elif authentication_status:
    # ⬇️ Oldalcím és kijelentkezés gomb
    st.set_page_config(page_title="ReflectAI")
    authenticator.logout('Kijelentkezés', 'oldalsáv')
    st.sidebar.title(f"Üdv, {name}!")
    st.title("ReflectAI modulválasztó")

    # ⬇️ Modul kiválasztása
    module_name = st.sidebar.radio("Válassz modult:", list(MODULES.keys()))

    # ⬇️ Modul futtatása biztonságosan
    safe_run(module_name)
