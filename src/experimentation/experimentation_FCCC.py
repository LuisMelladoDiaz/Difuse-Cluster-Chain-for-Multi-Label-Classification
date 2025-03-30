from models.FCCC import FCCC
from experimentation_framework import ExperimentationFramework

# CONFIGURACIÓN
DATASETS = {"Birds": 19, "Emotions": 6}
NUM_CLUSTERS = 3
NUM_EXPERIMENTOS = 20
THRESHOLD = 0.1

# Inicializar y ejecutar experimentos
framework = ExperimentationFramework(
    model_name="FCCC",
    model_function=FCCC,
    output_dir="experimentos/FCCC",
    datasets=DATASETS,
    num_experiments=NUM_EXPERIMENTOS,
    num_clusters=NUM_CLUSTERS,
    threshold=THRESHOLD
    
)

framework.run_experiments()
framework.save_results()
