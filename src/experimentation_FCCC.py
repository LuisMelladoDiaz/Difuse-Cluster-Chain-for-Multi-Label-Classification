from models.FCCC import FCCC
from experimentation_framework import ExperimentationFramework

# CONFIGURACIÓN
#DATASETS = {"Foodtruck": 12, "CHD_49": 6, "Water-quality": 14, "Birds": 19, "Emotions": 6,}
#DATASETS = {"Corel5k": 374}
#DATASETS = {"Mediamill": 101}
DATASETS = {"Yahoo_Business": 30}

NUM_CLUSTERS = 3
NUM_EXPERIMENTOS = 30
THRESHOLD = 0.05
NUM_FOLDS = 5

# Inicializar y ejecutar experimentos
framework = ExperimentationFramework(
    model_name="FCCC",
    model_function=FCCC,
    output_dir="experimentos/FCCC",
    datasets=DATASETS,
    num_experiments=NUM_EXPERIMENTOS,
    num_folds=NUM_FOLDS,
    num_clusters=NUM_CLUSTERS,
    threshold=THRESHOLD
)

framework.run_experiments()
framework.save_results()
