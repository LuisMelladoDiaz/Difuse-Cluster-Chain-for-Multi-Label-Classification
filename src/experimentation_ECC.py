from models.CC import train_ECC
from experimentation_framework import ExperimentationFramework

# CONFIGURACIÓN
DATASETS = {"Birds": 19, "Emotions": 6}
NUM_CHAINS = 10
NUM_EXPERIMENTOS = 10

# Inicializar y ejecutar experimentos
framework = ExperimentationFramework(
    model_name="ECC",
    model_function=train_ECC,
    output_dir="experimentos/ECC",
    datasets=DATASETS,
    num_experiments=NUM_EXPERIMENTOS,
    number_of_chains=NUM_CHAINS
)

framework.run_experiments()
framework.save_results()
