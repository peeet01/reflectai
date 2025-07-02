from modules.kuramoto_sim import run as run_kuramoto
from modules.hebbian_learning import run as run_hebbian
from modules.xor_prediction import run as run_xor
from modules.kuramoto_hebbian_sim import run as run_kuramoto_hebbian
from modules.topo_protect import run as run_topo_protect
from modules.lorenz_sim import run as run_lorenz_sim
from modules.mlp_predict_lorenz import run as run_lorenz_pred
from modules.berry_curvature import run as run_berry
from modules.noise_robustness import run as run_noise
from modules.esn_prediction import run as run_esn
from modules.plasticity_dynamics import run as run_plasticity
from modules.fractal_dimension import run as run_fractal
from modules.persistent_homology import run as run_homology
from modules.lyapunov_spectrum import run as run_lyapunov
from modules.memory_landscape import run as run_memory_landscape
from modules.graph_sync_analysis import run as run_graph_sync
from modules.generative_kuramoto import run as run_generative_kuramoto
from modules.insight_learning import run as run_insight
from modules.data_upload import run as run_data_upload
from modules.help_module import run as run_help
from modules.reflection_modul import run as run_reflection
from modules.questions import run as run_questions
from modules.neural_entropy import run as neural_entropy
from modules.criticality_explorer import app as run_criticality_explorer
from modules.oja_learning import run as run_oja_learning
from modules.stdp_learning import run as run_stdp_learning

# Modulregisztráció (név -> függvény)
registry = {
    "Kuramoto szimuláció": run_kuramoto,
    "Hebbian tanulás": run_hebbian,
    "XOR predikció": run_xor,
    "Kuramoto–Hebbian háló": run_kuramoto_hebbian,
    "Topológiai szinkronizáció": run_topo_protect,
    "Lorenz szimuláció": run_lorenz_sim,
    "Lorenz predikció (MLP)": run_lorenz_pred,
    "Berry-görbület": run_berry,
    "Zajtűrés": run_noise,
    "ESN predikció": run_esn,
    "Plaszticitás dinamikája": run_plasticity,
    "Fraktál dimenzió": run_fractal,
    "Perzisztens homológia": run_homology,
    "Lyapunov spektrum": run_lyapunov,
    "Memória tájkép": run_memory_landscape,
    "Gráf szinkron analízis": run_graph_sync,
    "Generatív Kuramoto": run_generative_kuramoto,
    "Belátás alapú tanulás": run_insight,
    "Adatfeltöltés": run_data_upload,
    "Súgó / Help": run_help,
    "Reflexió modul": run_reflection,
    "Kérdés-válasz modul": run_questions,
    "Oja Learning": run_oja,
    "Criticality explorer": run_criticality,
    "Neural Entropy": neural_entropy,
    "STDP tanulás": run_stdp,
}
