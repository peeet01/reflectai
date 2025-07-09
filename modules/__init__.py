from modules.kuramoto_sim import app as run_kuramoto
from modules.hebbian_learning import app as run_hebbian
from modules.xor_prediction import app as run_xor
from modules.kuramoto_hebbian_sim import app as run_kuramoto_hebbian
from modules.topo_protect import app as run_topo_protect
from modules.lorenz_sim import app as run_lorenz_sim
from modules.mlp_predict_lorenz import app as run_lorenz_pred
from modules.berry_curvature import app as run_berry
from modules.noise_robustness import app as run_noise_robustness
from modules.esn_prediction import app as run_esn
from modules.plasticity_dynamics import app as run_plasticity
from modules.fractal_dimension import app as run_fractal_dimension
from modules.persistent_homology import app as run_homology
from modules.lyapunov_spectrum import app as run_lyapunov
from modules.memory_landscape import app as run_memory_landscape
from modules.graph_sync_analysis import app as run_graph_sync
from modules.generative_kuramoto import app as run_generative_kuramoto
from modules.insight_learning import app as run_insight
from modules.data_upload import app as run_data_upload
from modules.help_module import app as run_help
from modules.reflection_modul import app as run_reflection
from modules.questions import app as run_questions
from modules.neural_entropy import app as neural_entropy
from modules.criticality_explorer import app as run_criticality_explorer
from modules.oja_learning import app as run_oja_learning
from modules.stdp_learning import app as run_stdp_learning
from modules.bcm_learning import app as run_bcm_learning
from modules.snn_simulation import app as run_snn
from modules.fractal_explorer import app as run_fractal_explorer
from modules.ising_sim import app as run_ising_sim
from modules.hebbian_learning_viz import app as run_hebbian_viz
from modules.critical_hebbian import app as run_critical_hebbian

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
    "Zajtűrés": run_noise_robustness,
    "ESN predikció": run_esn,
    "Plaszticitás dinamikája": run_plasticity,
    "Fraktál dimenzió": run_fractal_dimension,
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
    "Oja Learning": run_oja_learning,
    "Criticality explorer": run_criticality,
    "Neural Entropy": neural_entropy,
    "STDP tanulás": run_stdp,
    "BCM tanulás": run_bcm_learning,
    "Spiking Neural Network": run_snn,
    "Fraktál Explorer": run_fractal_explorer,
    "Ising szimulácio": run_ising_sim,
    "Hebbian Learning Viz": run_hebbian_viz,
    "Critical Hebbian tanulás (3D)": run_critical_hebbian,
}
