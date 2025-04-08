from models.CC import train_CC
from experimentation_framework import ExperimentationFramework

# CONFIGURACIÓN
#DATASETS = {"Foodtruck": 12, "CHD_49": 6, "Water-quality": 14, "Birds": 19, "Emotions": 6}
DATASETS = {"Yahoo_Business": 30}

NUM_EXPERIMENTOS = 30
NUM_FOLDS = 5


# EXPERIMENTAR

framework = ExperimentationFramework(
    model_name="CC",
    model_function=train_CC,
    output_dir="experimentos/CC",
    datasets=DATASETS,
    num_experiments=NUM_EXPERIMENTOS,
    num_folds=NUM_FOLDS

)

framework.run_experiments()
framework.save_results()
