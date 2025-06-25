import streamlit as st
import yaml
import streamlit_authenticator as stauth
from pathlib import Path
from modules import help_module
from modules import questions
from modules import reflection_modul
from modules import insight_learning
from modules import hebbian_learning_viz
from modules import graph_sync_analysis
from modules import persistent_homology
from modules import plasticity_dynamics

# ----- Hitelesítés -----
with open(Path(".") / "config.yaml") as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config.get("preauthorized", {})
)

# 🔐 Bejelentkezés – új API szerint
name, authentication_status, username = authenticator.login(
    fields={"Form name": "Bejelentkezés"},
    location="main"
)

if authentication_status is False:
    st.error("❌ Hibás felhasználónév vagy jelszó.")
elif authentication_status is None:
    st.warning("⚠️ Kérlek add meg a bejelentkezési adataidat.")
elif authentication_status:
    authenticator.logout("Kijelentkezés", "sidebar")
    st.sidebar.success(f"✅ Bejelentkezve mint: {name}")

    # ----- Menü -----
    st.sidebar.title("Navigáció")
    oldalak = {
        "ℹ️ Súgó": help_module.main,
        "❓ Kérdések": questions.main,
        "🧠 Reflexió": reflection_modul.main,
        "💡 Insight learning": insight_learning.main,
        "🔁 Hebbian tanulás": hebbian_learning_viz.main,
        "🔗 Gráf szinkron": graph_sync_analysis.main,
        "🏔️ Perzisztens homológia": persistent_homology.main,
        "📈 Plaszticitás": plasticity_dynamics.main,
    }

    valasztott = st.sidebar.radio("Válassz modult:", list(oldalak.keys()))
    oldal_fuggveny = oldalak[valasztott]
    oldal_fuggveny()
