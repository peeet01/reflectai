import streamlit as st

def run():
    st.title("📚 Segítség és modulismertető")
    st.markdown("""
Ez az alkalmazás különféle **neurális és komplex rendszermodelleket** demonstrál. Minden modul célja a biológiai, fizikai vagy mesterséges intelligencia rendszerek egyes dinamikus tulajdonságainak szemléltetése.

---

### 🧠 Hebbian tanulás
**Cél:** Megmutatni a Hebb-féle tanulási szabályt: *"Cells that fire together, wire together."*  
**Egyenlet:**  
\
\\Delta w_{ij} = \\eta x_i y_j
\  
ahol \w_{ij}\ a szinaptikus súly, \x_i\ a bemenet, \y_j\ a kimenet, és \\\eta\ a tanulási ráta.

---

### 🤖 XOR predikció
**Cél:** Egy neurális háló tanítása az XOR logikai kapu megtanulására.  
**Tudományos háttér:** Az XOR probléma nemlineárisan szeparálható, ezért szükséges rejtett réteg (MLP).  
**Adatfeltöltés támogatott:** Igen.

---

### 🔗 Kuramoto szinkronizáció
**Cél:** Oszcillátorok közötti fázisszinkronizáció modellezése.  
**Egyenlet:**  
\
\\frac{d\\theta_i}{dt} = \\omega_i + \\frac{K}{N} \\sum_{j=1}^{N} \\sin(\\theta_j - \\theta_i)
\  
ahol \\\theta_i\ az oszcillátor fázisa, \\\omega_i\ a sajátfrekvencia, és \K\ a csatolás.

---

### 🧠 Kuramoto–Hebbian hálózat
**Cél:** Kuramoto és Hebbian dinamikák egyesítése a tanuló szinkronizációs hálózathoz.  
**Hatás:** Biológiai szinkronizáció és adaptív tanulás modellezése.

---

### 🌐 Topológiai szinkronizáció
**Cél:** Hálózati topológia hatásának vizsgálata Kuramoto-modellek szinkronizációjára.  
**Moduláris:** Egyedi gráfstruktúrák is tesztelhetők.

---

### 🌪️ Lorenz szimuláció
**Cél:** Káoszos Lorenz-rendszer szimulációja.  
**Egyenletek:**  
\
\\begin{cases}
\\dot{x} = \\sigma(y - x) \\\\
\\dot{y} = x(\\rho - z) - y \\\\
\\dot{z} = xy - \\beta z
\\end{cases}
\  
**Tudományos háttér:** Időjárásmodellezés és determinisztikus káosz.

---

### 🔮 Lorenz predikció (MLP)
**Cél:** Többrétegű perceptron tanítása Lorenz-idősor előrejelzésére.  
**Adatfeltöltés támogatott:** Igen.

---

### 🔍 Lorenz predikció (ESN)
**Cél:** Echo State Network alkalmazása időbeli predikcióra Lorenz adatokon.  
**Tudományos háttér:** Rezonáns tartománytanulás.  
**Adatfeltöltés támogatott:** Igen.

---

### 🔊 Zajtűrés és szinkronizációs robusztusság
**Cél:** Kuramoto-modell vizsgálata zajos környezetben.  
**Fő mérőszám:** Átlagos szinkronizációs index \r\.  
**Adatfeltöltés támogatott:** Igen.

---

### 🧮 Topológiai védettség (Chern-szám)
**Cél:** Szilárdtestfizikából ismert topológiai fázisok Chern-számának kiszámítása.  
**Alkalmazás:** Kvantumos Hall-hatás, topológiai szigetelők.  
**Matematika:** Integrál a Berry-görbületen.

---

### 🔄 Topológiai Chern-szám analízis
**Cél:** A Berry-görbület és a Chern-szám numerikus számítása adott mátrixok alapján.  
**Fizikai jelentőség:** Diszkrét rácsmodellek topológiai tulajdonságai.

---

### 🧪 Hebbian plaszticitás dinamikája
**Cél:** A tanulási folyamatok hosszútávú stabilitásának vizsgálata Hebbian alapján.  
**Hatás:** Túlillesztés és divergens súlyok elemzése.

---

### 📐 Szinkronfraktál dimenzióanalízis
**Cél:** Fázisszinkronizációból származó adatok fraktáldimenziójának becslése.  
**Módszer:** Box-counting algoritmus.  
**Adatfeltöltés támogatott:** Igen.

---

### 👁️‍🗨️ Belátás alapú tanulás (Insight Learning)
**Cél:** Problémamegoldás szimulálása belső reprezentáció alapján.  
**Tudományos háttér:** Köhler majomkísérletei, Gestalt pszichológia.

---

### 🧬 Generatív Kuramoto-hálózat
**Cél:** Dinamikus gráfgenerálás és Kuramoto szinkronizáció összeolvasztása.  
**Használat:** Rendszergenerálás és vizuális szinkronállapotok.

---

### 🧠 Memória tájkép (Memory Landscape)
**Cél:** A tárolt emlékek közötti energiatájak vizualizálása.  
**Inspiráció:** Hopfield-hálózatok és memóriadinamika.

---

### 🧩 Gráfalapú szinkronanalízis
**Cél:** Általános szinkronanalízis gráfstruktúrák és csatolási erősségek alapján.

---

### 🌀 Lyapunov-spektrum
**Cél:** A Lorenz-rendszer legnagyobb Lyapunov-exponensének becslése.  
**Matematika:**  
\
\\lambda = \\lim_{t \\to \\infty} \\frac{1}{t} \\sum_{i=1}^{t} \\log \\left( \\frac{\\|\\delta(t+1)\\|}{\\|\\delta(t)\\|} \\right)
\  
**Adatfeltöltés támogatott:** Tervezett.

---

""")
