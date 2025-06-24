import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Load credentials
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days'],
    preauthorized=config.get('preauthorized', {})
)

# Login
name, authentication_status, username = authenticator.login('Bejelentkezés', location='main')

if authentication_status:
    authenticator.logout('Kijelentkezés', 'sidebar')
    st.sidebar.title(f'Üdvözlünk, {name}!')
    st.title('ReflectAI')
    st.write('Ez a kezdőlap. Itt jelenik meg az app többi funkciója.')
elif authentication_status is False:
    st.error('Hibás felhasználónév vagy jelszó.')
elif authentication_status is None:
    st.warning('Kérlek jelentkezz be.')
