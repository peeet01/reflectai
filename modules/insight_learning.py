import streamlit as st import numpy as np import matplotlib.pyplot as plt import time import random

def run(trials=5, pause_time=1.0, complexity="kozepes"): st.write("### 🧠 Belátás alapú tanulás – Szimuláció")

# Komplexitás szintjeihez paraméterek
if complexity == "alacsony":
    n_elements = 3
elif complexity == "magas":
    n_elements = 7
else:
    n_elements = 5

fig, ax = plt.subplots()
progress_bar = st.progress(0)
log_area = st.empty()

success_found = False
insight_trial = random.randint(2, trials)  # belátás várhatóan itt következik be
solution_path = np.sort(np.random.permutation(range(1, 10))[:n_elements])

log = ""
for t in range(1, trials + 1):
    progress_bar.progress(t / trials)

    ax.clear()
    attempt = np.sort(np.random.permutation(range(1, 10))[:n_elements])
    ax.bar(range(n_elements), attempt)
    ax.set_title(f"{t}. próbálkozás – Elemkombináció")
    st.pyplot(fig)

    if np.array_equal(attempt, solution_path) and t >= insight_trial:
        log += f"\n✅ {t}. próbálkozás: Sikeres belátás! Megfejtett struktúra: {attempt}"
        success_found = True
        break
    else:
        log += f"\n❌ {t}. próbálkozás: Nem sikerült. Próbálkozott: {attempt}"
        time.sleep(pause_time)
    log_area.code(log)

if not success_found:
    log += f"\n⚠️ {trials}. próbálkozás után sem történt áttörés."
    log_area.code(log)
else:
    st.balloons()
    st.success("Belátás megtörtént!")

st.markdown("---")
st.markdown("#### 🔎 Megjegyzés")
st.info("A belátás szimulációja során egy rejtett mintázatot próbál meg felfedezni a modell próbálkozások sorozatával. A 'megoldás' akkor érkezik meg, ha egy belső áttörés történik – ez itt egy véletlenszerű kísérlet, ami az előre definiált mintára illeszkedik.")

