import streamlit as st

def run(): st.title("❓ Súgó / Modulismertető") st.markdown(""" Itt megtalálod az egyes modulok részletes leírását, célját, tudományos hátterét, valamint a használt matematikai egyenleteket.

---

### 🧭 Kuramoto szinkronizáció
**Cél:** Vizsgálni, hogyan képesek oszcillátorok szinkronizálódni csatolás hatására.  
**Háttér:** A Kuramoto-modell az egyik legismertebb modell fázis-szinkronizációra komplex rendszerekben.  
**Egyenlet:**  
$\dot{\theta}_i = \omega_i + \frac{K}{N} \sum_{j=1}^N \sin(\theta_j - \theta_i)$

---

### 🧠 Hebbian tanulás
**Cél:** Neurális tanulási szabály szimulálása, miszerint együtt aktív neuronok kapcsolatai erősödnek.  
**Háttér:** Donald Hebb elmélete szerint a tanulás az együttes aktivitás alapján történik.  
**Szabály:**  
$\Delta w_{ij} = \eta x_i y_j$

---

### ❌ XOR predikció
**Cél:** Megtanítani egy egyszerű neurális hálót egy nemlineáris logikai műveletre.  
**Háttér:** Az XOR probléma nem oldható meg egyetlen lineáris perceptronnal – ez vezette az MLP fejlődését.  
**Tanítás:** Két bináris bemenetből egy bináris kimenet tanulása MLP-vel.

---

### 🌐 Kuramoto–Hebbian hálózat
**Cél:** Kombinálni a fázisszinkronizációt és a Hebb-féle tanulást.  
**Háttér:** Dinamikus gráfháló, ahol a kapcsolat erőssége időben változik tanulás hatására.

---

### 🧩 Topológiai szinkronizáció
**Cél:** Megérteni, hogyan befolyásolja a hálózat struktúrája a szinkronizációt.  
**Háttér:** A gráf topológiája kulcsfontosságú tényező a kollektív dinamika alakulásában.

---

### 🌪️ Lorenz szimuláció
**Cél:** A Lorenz-rendszer numerikus integrálása, és káoszos viselkedés megfigyelése.  
**Háttér:** Meteorológiai eredetű, három differenciálegyenletből álló nemlineáris rendszer.  
**Egyenletek:**  
$\dot{x} = \sigma(y - x)$  
$\dot{y} = x(\rho - z) - y$  
$\dot{z} = xy - \beta z$

---

### 🔮 Lorenz predikció (MLP/ESN)
**Cél:** A Lorenz-rendszer idősoraiból a jövőbeli értékek előrejelzése.  
**Háttér:** Idősor-előrejelzés nemlineáris dinamikus rendszereken.  
**Modellek:** Többrétegű perceptron (MLP) és Echo State Network (ESN).  
**ESN formula:**  
$x(t+1) = \tanh(W_{in}u(t) + Wx(t))$

---

### 🛡️ Topológiai védettség (Chern-szám)
**Cél:** Hálózati topológiai jellemzők (Chern-szám) vizsgálata robusztusság szempontjából.  
**Háttér:** Kvantum Hall-effektus analógiája diszkrét gráfokban.  
**Mennyiség:** Topológiai invariáns:  
$C = \frac{1}{2\pi} \int_{BZ} \Omega(k) \, d^2k$

---

### 🔢 Chern–szám analízis
**Cél:** Chern-szám numerikus számítása Berry-görbület alapján.  
**Háttér:** Kvantummechanikai hullámfüggvény geometriai fázisa.

---

### 🔊 Zajtűrés / Noise robustness
**Cél:** Kuramoto-szinkronizációs viselkedés vizsgálata különböző zajszinteken.  
**Háttér:** Valós rendszerekben a szinkronizáció stabilitását zaj befolyásolja.

---

### 🧠 Echo State Network (ESN)
**Cél:** Nemlineáris idősorok előrejelzése sztochasztikus reservoirok segítségével.  
**Háttér:** A belső dinamikát nem tanítjuk, csak a kimeneti lineáris olvasó réteget.  
**Egyenlet:**  
$x(t+1) = \tanh(W_{in} u(t) + Wx(t))$

---

### 🔁 Hebbian plaszticitás
**Cél:** A tanulási szabály időbeli dinamikájának modellezése.  
**Háttér:** Biológiailag motivált súlyváltozási törvények időfüggéssel.

---

### 🌀 Szinkronfraktál dimenzióanalízis
**Cél:** Fraktáldimenzió meghatározása idősor ponteloszlásból.  
**Háttér:** A fraktáldimenzió a rendszer komplexitását jellemzi.  
**Becslés:**  
$D \approx \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log(1/\epsilon)}$

---

### 💡 Insight learning
**Cél:** Belátás alapú tanulási formák szimulációja.  
**Háttér:** A megértésen alapuló tanulás különbözik a klasszikus kondicionálástól.

---

### 🧬 Generatív Kuramoto hálózat
**Cél:** Strukturális fejlődés szimulációja Kuramoto-alapú hálóban.  
**Háttér:** Egyesíti a gráfépítést és tanulást szinkronizációval.

---

### 🧠 Memória tájkép
**Cél:** Rekurrens hálózatok stabil állapotainak és memóriájának feltérképezése.  
**Háttér:** Energiaalapú modellezés (pl. Hopfield-hálók).

---

### 🧩 Gráfalapú szinkronanalízis
**Cél:** Komplex hálózat szinkronizációs tulajdonságainak vizsgálata topológiai függvényében.

---

### 📉 Lyapunov spektrum
**Cél:** A káosz mérőszámának (legnagyobb Lyapunov-exponens) becslése Lorenz-pályából.  
**Háttér:** Egy kis perturbáció időbeli növekedésének logaritmikus mértéke.  
**Képlet:**  
$\lambda = \lim_{t \to \infty} \frac{1}{t} \log \frac{\|\delta(t)\|}{\|\delta(0)\|}$

---

### 📁 Adatfeltöltés modul
**Cél:** CSV fájlok feltöltése, előnézete, validálása.  
**Funkció:** Session-ben tárolás, oszlopellenőrzés, fallback adatok kezelése.

---

Ha kérdésed van, fordulj bizalommal a fejlesztőhöz! 📬
""")

