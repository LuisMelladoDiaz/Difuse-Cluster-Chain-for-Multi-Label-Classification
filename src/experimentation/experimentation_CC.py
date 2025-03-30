from models.CC import train_CC
from experimentation_framework import ExperimentationFramework

# CONFIGURACIÓN
DATASETS = {"Birds": 19, "Emotions": 6}
NUM_EXPERIMENTOS = 10

# EXPERIMENTAR

framework = ExperimentationFramework(
    model_name="CC",
    model_function=train_CC,
    output_dir="experimentos/CC",
    datasets=DATASETS,
    num_experiments=NUM_EXPERIMENTOS
)

framework.run_experiments()
framework.save_results()
