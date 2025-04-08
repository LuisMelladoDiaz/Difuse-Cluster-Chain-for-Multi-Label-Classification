from models.CC import train_ECC
from experimentation_framework import ExperimentationFramework

# CONFIGURACIÓN
DATASETS = {"Foodtruck": 12, "CHD_49": 6, "Water-quality": 14, "Birds": 19, "Emotions": 6,}
NUM_CHAINS = 10
NUM_EXPERIMENTOS = 30
NUM_FOLDS = 5

# Inicializar y ejecutar experimentos
framework = ExperimentationFramework(
    model_name="ECC",
    model_function=train_ECC,
    output_dir="experimentos/ECC",
    datasets=DATASETS,
    num_experiments=NUM_EXPERIMENTOS,
    number_of_chains=NUM_CHAINS,
    num_folds=NUM_FOLDS
)

framework.run_experiments()
framework.save_results()
