from models.FCCC import FCCC
from experimentation_framework import ExperimentationFramework
from models.LCC_MLC import LCC_MLC

# CONFIGURACIÓN
DATASETS = {"Foodtruck": 12, "CHD_49": 6, "Water-quality": 14, "Birds": 19, "Emotions": 6,}

NUM_CLUSTERS = 3
NUM_EXPERIMENTOS = 30
THRESHOLD = 0.05
NUM_FOLDS = 5

# Inicializar y ejecutar experimentos
framework = ExperimentationFramework(
    model_name="LCC-MLC",
    model_function=LCC_MLC,
    output_dir="experimentos/LCC-MLC",
    datasets=DATASETS,
    num_experiments=NUM_EXPERIMENTOS,
    num_folds=NUM_FOLDS
)

framework.run_experiments()
framework.save_results()
