import streamlit as st

def run():
    st.title("❓ Súgó – Neurolab AI Scientific Playground Sandbox")

    st.markdown("""
    ## 🔍 Mi ez az alkalmazás?
    A **Neurolab AI** egy nyílt kutatásorientált interaktív sandbox, amely lehetővé teszi különböző mesterséges intelligencia modellek, dinamikai rendszerek és hálózati szimulációk futtatását és megértését. A cél, hogy **kutatók, hallgatók, oktatók és fejlesztők** számára egy szemléletes, moduláris és bővíthető felület álljon rendelkezésre a gépi tanulás, idegrendszeri dinamika és szinkronizáció területein.

    ---

    ## 🧭 Modulismertető (Tudományos leírásokkal)

    ### 🔁 XOR predikció neurális hálóval
    - **Cél:** Egy bináris logikai függvény (XOR) megtanítása egy több rétegű perceptron segítségével.
    - **Tudományos háttér:** Az XOR nemlineáris problémát jelent, amit egyetlen rétegű háló nem tud megtanulni, de egy rejtett réteggel rendelkező MLP képes rá. A modul a tanulási folyamatot vizsgálja zajos bemeneti adatokkal és visszacsatolással.
    
    ### 🧭 Kuramoto szinkronizáció
    - **Cél:** Az oszcillátorok kollektív szinkronizációs viselkedésének modellezése.
    - **Tudományos háttér:** A Kuramoto-modell egy klasszikus nemlineáris differenciálegyenlet-rendszer, ami fázisoszcillátorok közötti szinkronizációt ír le. Alkalmas idegi szinkronizáció, biológiai ritmusok vagy hálózati koherencia vizsgálatára.

    ### 🧠 Hebbian tanulás
    - **Cél:** A Hebb-féle tanulási szabály szemléltetése.
    - **Tudományos háttér:** A „neurons that fire together wire together” elv alapján a neuronkapcsolatok erősödnek, ha az aktivációjuk korrelál. Ez az alapja a szinaptikus plaszticitásnak, a hosszú távú memóriaképzésnek.

    ### ⚡ Kuramoto–Hebbian hálózat
    - **Cél:** Dinamikus szinkronizációs és adaptív súlytanulási folyamatok kombinációja.
    - **Tudományos háttér:** A Kuramoto fázismodell Hebbian tanulással való összekapcsolása bemutatja, hogyan fejlődhet a hálózati konnektivitás a kollektív dinamika hatására.

    ### 🔒 Topológiai szinkronizáció
    - **Cél:** A hálózati struktúra hatása a szinkronizációs dinamika stabilitására.
    - **Tudományos háttér:** A szinkronizáció stabilitását nagyban befolyásolja a gráf topológiája. A szimuláció azt vizsgálja, hogy különböző topológiák hogyan hatnak a koherenciára.

    ### 🌀 Lorenz rendszer (szimuláció)
    - **Cél:** A determinisztikus káosz bemutatása.
    - **Tudományos háttér:** A Lorenz-rendszer az időjárás előrejelzésének egyik modellje, amelyet Edward Lorenz dolgozott ki. Nemlineáris, determinisztikus, de kaotikus viselkedést mutat.

    ### 🔮 Lorenz predikció
    - **Cél:** Neurális háló alkalmazása kaotikus rendszer előrejelzésére.
    - **Tudományos háttér:** Idősor predikció mélytanulással, a nemlineáris dinamikai rendszerek tanulmányozásához.

    ### 🧬 Zajtűrés és szinkronizációs robusztusság
    - **Cél:** A szinkronizáció érzékenységének mérése külső zajra.
    - **Tudományos háttér:** Egy rendszer zajtűrésének vizsgálata elengedhetetlen a valós adatokkal történő alkalmazásokhoz, különösen idegi hálók és fizikai rendszerek esetén.

    ### 🧩 Topológiai Chern–szám analízis
    - **Cél:** Topológiai invariánsok numerikus meghatározása.
    - **Tudományos háttér:** A Chern-szám kvantált topológiai mennyiség, amely a Berry-görbület integráljaként jelenik meg a kvantumfizikában és topologikus anyagokban.

    ### 🧠 Belátás alapú tanulás (Insight Learning)
    - **Cél:** Tanulási szimuláció, ahol a megoldás hirtelen jelenik meg – nem fokozatos tanulás eredménye.
    - **Tudományos háttér:** A Gestalt-pszichológiából eredő modell, amely bemutatja, hogy a megértés nem mindig tapasztalaton alapuló próbálkozás.

    ### 📈 Echo State Network (ESN) predikció
    - **Cél:** Dinamikus rendszerek memóriaalapú előrejelzése visszacsatolt hálóval.
    - **Tudományos háttér:** A Recurrent Neural Network (RNN) egy típusa, amely fixen inicializált rejtett állapotokat használ, és csak a kimeneti súlyokat tanítja.

    ### 🔄 Hebbian plaszticitás dinamikája
    - **Cél:** Szinaptikus súlyváltozások vizsgálata időben.
    - **Tudományos háttér:** A hosszú távú potenciáció (LTP) és depresszió (LTD) modellezése Hebbian mechanizmus alapján.

    ### 🧮 Szinkronfraktál dimenzióanalízis
    - **Cél:** A fázisszinkronizáció alapján képzett fraktálstruktúrák dimenziójának mérése.
    - **Tudományos háttér:** A szinkronizáció mintázatainak fraktálszerkezete kulcsfontosságú lehet komplex rendszerek elemzésében.

    ### 🧠 Generatív Kuramoto hálózat
    - **Cél:** Dinamikusan generált gráfstruktúrák Kuramoto-alapú szinkronizációs vizsgálata.
    - **Tudományos háttér:** Véletlenszerűen épülő oszcillátorhálózatok szinkronizációs tulajdonságainak feltérképezése.

    ### 🧭 Memória tájkép (Memory Landscape)
    - **Cél:** Memóriaállapotok feltérképezése neurális rendszerekben.
    - **Tudományos háttér:** Az állapottér topográfiája hatással van a memória stabilitására és hozzáférhetőségére.

    ---

    ## 📦 Export és mentés
    - CSV export predikciós eredményekhez
    - Modellmentés `.pth` fájlba újrabetöltéshez
    - Jegyzetmentés a vizsgálatok dokumentálásához

    ---

    ## 👥 Célközönség
    - **Kutatók:** Elméleti modellek gyors verifikálása
    - **Oktatók:** Interaktív szemléltető eszközök
    - **Diákok:** Vizsgálati és tanulási lehetőség mélytanuláshoz
    - **Fejlesztők:** Nyílt és bővíthető architektúra kipróbálása
    """)
