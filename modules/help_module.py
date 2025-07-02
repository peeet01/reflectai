import streamlit as st

def run():
    st.title("❓ Súgó és Dokumentáció – Neurolab AI")
    st.markdown("""
Üdvözölünk a **Neurolab AI Sandbox** környezetben!  
Ez az alkalmazás idegtudományi, hálózatelméleti és tanulási modelleket tesztel, demonstrál és dokumentál.
""")

    def add_section(title, description_md, refs=None):
        st.markdown(f"---\n### {title}")
        st.markdown(description_md)
        if refs:
            st.markdown("**Források:**")
            for text, src in refs:
                st.markdown(f"- [{text}]({src})")

    # Kuramoto Modell
    add_section(
        "🕸️ Kuramoto Modell – Szinkronizációs Dinamika",
        r"""
$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^{N} A_{ij} \sin(\theta_j - \theta_i)$

**Jelölések**:  
- $\theta_i$: az *i*-edik oszcillátor fázisa  
- $\omega_i$: természetes frekvencia  
- $K$: kapcsolódási erősség  
- $A_{ij}$: kapcsolódási mátrix  
- $N$: oszcillátorok száma

A Kuramoto-modell kollektív viselkedést modellez komplex rendszerekben (pl. agyhullámok, biológiai ritmusok).
""",
        refs=[
            ("Strogatz 2000 – From Kuramoto to Collective Behavior", "https://doi.org/10.1063/1.2781371"),
            ("Nature Scientific Reports 2019", "https://www.nature.com/articles/s41598-019-54769-9")
        ]
    )

    # XOR modell
    add_section(
        "❌ XOR Predikció – Többrétegű Perceptron (MLP)",
        r"""
$\hat{y} = \sigma\left( W^{(2)} \cdot \sigma(W^{(1)} \cdot x + b^{(1)}) + b^{(2)} \right)$  
$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$

Ahol $\sigma(x)$ a szigmoid aktivációs függvény.

Ez az alapja a mélytanulás fejlődésének, hiszen lineáris modellek nem képesek megoldani az XOR-problémát.
""",
        refs=[
            ("MLP and XOR separability – classic AI challenge", "https://cs231n.github.io/neural-networks-1/")
        ]
    )

    # Berry-görbület
    add_section(
        "🌐 Berry-görbület – Topológiai Kvantumfizika",
        r"""
$\Omega(\mathbf{k}) = \nabla_{\mathbf{k}} \times \mathbf{A}(\mathbf{k})$  
$\mathbf{A}(\mathbf{k}) = -i \langle u(\mathbf{k}) | \nabla_{\mathbf{k}} | u(\mathbf{k}) \rangle$

A Berry-görbület kvantummechanikai topológiai struktúrákat ír le – fontos a kvantum Hall-effektus és topológiai szigetelők esetén.
""",
        refs=[
            ("Xiao et al., Rev. Mod. Phys. 82, 1959 (2010)", "https://doi.org/10.1103/RevModPhys.82.1959"),
            ("PRL 134.016601 (2025)", "https://doi.org/10.1103/PhysRevLett.134.016601")
        ]
    )

    # Hopfield-háló
    add_section(
        "🧠 Hopfield-háló – Asszociatív Memória",
        r"""
$W_{ij} = \sum_{\mu=1}^P \xi_i^\mu \xi_j^\mu,\quad W_{ii} = 0$  
$s_i^{t+1} = \mathrm{sign} \left( \sum_j W_{ij} s_j^t \right)$

A háló stabil állapotokra konvergál, így mintatárolóként működik.

Modern változatai (Dense Associative Memory) képesek sokkal több mintát tárolni és pontosabban visszahívni.
""",
        refs=[
            ("Hopfield 1982", "https://doi.org/10.1073/pnas.79.8.2554"),
            ("Krotov & Hopfield (2016)", "https://arxiv.org/abs/1606.00918")
        ]
    )

    # Fraktál Explorer
    add_section(
        "🌀 Fraktál Explorer – Mandelbrot-halmaz",
        r"""
$z_{n+1} = z_n^2 + c$

A fraktálok a determinisztikus kaotikus rendszerek vizsgálatára alkalmasak.  
Ez a modul vizuálisan is bemutatja a komplex dinamikát.
""",
        refs=[
            ("Peitgen et al. – Chaos and Fractals", "https://doi.org/10.1007/978-1-4757-2675-3")
        ]
    )

    # Egyéb modulok
    add_section(
        "✅ További Modulok – Áttekintés és cél",
        """
- **ESN Prediction**: Echo State Network modellek időbeli sorozatok előrejelzésére  
- **Generative Kuramoto**: gráf-alapú oszcillátor szimulációk automatikus generálása  
- **Graph Sync Analysis**: gráfstruktúra és szinkron dinamikák kapcsolata  
- **Persistent Homology**: topológiai adatértelmezés perzisztens diagramokkal  
- **Memory Landscape**: Hopfield-háló energiafelszínek vizualizációja  
- **Reflection Modul**: saját megfigyelések, hipotézisek naplózása  
""",
        refs=[
            ("Neurolab AI Modules – GitHub Wiki", "https://github.com/neurolab-ai/modules/wiki")
        ]
    )

    # Bizonyítási ötletek
    add_section(
        "🧪 Bizonyítási ötletek és kutatási irányok",
        """
- A Kuramoto-modell gráfelméleti stabilitási kritériumai  
- XOR probléma: nemlinearitás hatása tanulhatóságra  
- Berry-görbület: Chern-szám invariancia és topológiai fázisok  
- Hopfield-háló: minimális energiaszintek, stabilitási tájkép vizsgálata
"""
    )

    st.markdown("---")
    st.markdown("Verzió: **2025.07**  \nKészítette: *Koacs Peti*")

# ReflectAI modul belépési pont
app = run
